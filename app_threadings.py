import argparse
import cv2
import os
import time
import json
import logging
import base64
from flask import Flask, render_template, Response, jsonify
from edge_impulse_linux.image import ImageImpulseRunner
import threading

app = Flask(__name__, static_folder='templates/assets')

# Ensure the output directory exists
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ImageProcessor:
    def __init__(self, camera_id, object_size, save_images_interval, desired_fps=30, scale_factor=6):
        self.camera_id = camera_id
        self.object_size = object_size
        self.save_images_interval = save_images_interval
        self.desired_fps = desired_fps
        self.scale_factor = scale_factor

        # Initialize variables
        self.latest_high_res_frame = None
        self.inference_frame = None
        self.bounding_boxes = []
        self.extracted_objects = []
        self.count_objects = 0
        self.inference_speed = 0
        self.last_saved_time = 0
        self.od_model_parameters = None

        # Load model file paths
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.object_detection_model_file = os.path.join(dir_path, 'modelfile-fomo.eim')
        self.anomaly_detection_model_file = os.path.join(dir_path, 'modelfile-fomoad.eim')

        # Initialize runners
        self.object_detection_runner = ImageImpulseRunner(self.object_detection_model_file)
        self.anomaly_detection_runner = None  # Will be initialized when needed

        # Start threads
        self.start_threads()

    def start_threads(self):
        # Start high-res frame capture in a background thread
        self.capture_thread = threading.Thread(target=self.capture_high_res_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def capture_high_res_frames(self):
        # Capture high-res frames from camera or video file
        if os.path.isfile(self.camera_id):
            print(f"Opening video file: {self.camera_id}")
            camera = cv2.VideoCapture(self.camera_id)
            is_video_file = True
        else:
            print(f"Using camera ID: {self.camera_id}")
            camera = cv2.VideoCapture(int(self.camera_id))
            is_video_file = False

        delay_between_frames = 1 / self.desired_fps

        while True:
            start_time = time.time()
            success, frame = camera.read()
            if not success and is_video_file:
                print("End of video file, looping back to start.")
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            elif not success:
                print("Could not get frame (camera failure)")
                break

            # Store the latest frame
            self.latest_high_res_frame = frame.copy()

            # Sleep to maintain desired FPS
            frame_process_time = time.time() - start_time
            time_to_sleep = max(0, delay_between_frames - frame_process_time)
            time.sleep(time_to_sleep)

        camera.release()

    def process_frames(self):
        # Run object detection and anomaly detection
        with self.object_detection_runner as od_runner:
            try:
                model_info = od_runner.init()
                self.od_model_parameters = model_info['model_parameters']
                while True:
                    if self.latest_high_res_frame is None:
                        time.sleep(0.1)
                        continue

                    # Run object detection
                    self.run_object_detection(od_runner)

                    # Extract detected objects
                    self.extract_detected_objects()

                    # Run anomaly detection
                    self.run_anomaly_detection()

                    # Sleep briefly to avoid tight loop
                    time.sleep(0.01)
            finally:
                od_runner.stop()

    def run_object_detection(self, od_runner):
        img = cv2.cvtColor(self.latest_high_res_frame.copy(), cv2.COLOR_BGR2RGB)
        features, resized_image = od_runner.get_features_from_image(img)
        res = od_runner.classify(features)

        # Store the dimensions of the resized image
        self.resized_image_shape = resized_image.shape[:2]  # (height, width)

        if "result" in res:
            # Update inference_frame to be the resized image
            resized_image = cv2.resize(resized_image, (resized_image.shape[1] * self.scale_factor, resized_image.shape[0] * self.scale_factor))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            self.process_inference_result(res, resized_image)


    def process_inference_result(self, res, cropped):
        self.count_objects = 0
        self.bounding_boxes = []
        self.inference_speed = res['timing']['classification']
        if "bounding_boxes" in res["result"]:
            for bb in res["result"]["bounding_boxes"]:
                if bb['value'] > 0:
                    self.count_objects += 1
                    self.bounding_boxes.append({
                        'label': bb['label'],
                        'x': int(bb['x']),
                        'y': int(bb['y']),
                        'width': int(bb['width']),
                        'height': int(bb['height']),
                        'confidence': bb['value']
                    })
            self.inference_frame = self.draw_centroids(cropped, self.bounding_boxes)
        else:
            self.inference_frame = cropped

    def draw_centroids(self, frame, bounding_boxes):
        resized_height, resized_width = self.resized_image_shape
        inference_frame_height, inference_frame_width, _ = frame.shape

        scale_factor_x = inference_frame_width / resized_width
        scale_factor_y = inference_frame_height / resized_height

        for bb in bounding_boxes:
            center_x = int((bb['x'] + bb['width'] / 2) * scale_factor_x)
            center_y = int((bb['y'] + bb['height'] / 2) * scale_factor_y)
            frame = cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
            label_text = f"{bb['label']}: {bb['confidence']:.2f}"
            label_position = (int(bb['x'] * scale_factor_x), int((bb['y'] - 10) * scale_factor_y))
            cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def extract_detected_objects(self):
        self.extracted_objects = []
        squared_high_res_frame = self.get_squared_image_from_high_res_frame(self.latest_high_res_frame)
        for bb in self.bounding_boxes:
            
            cropped_img = self.crop_bounding_box(squared_high_res_frame, bb)
            cropped_img = cv2.resize(cropped_img, (self.object_size, self.object_size))
            self.extracted_objects.append({'bounding_box': bb, 'cropped_image': cropped_img})

            # Optionally save images
            self.save_cropped_image_if_needed(cropped_img, bb['label'])

    # Get a squared from the high resolution image. 
    # This will be used to map the coordinates of the bounding boxes.
    def get_squared_image_from_high_res_frame(self, frame):
        height, width, _ = frame.shape
        shortest_axis = min(height, width)
        x_start = (width - shortest_axis) // 2
        y_start = (height - shortest_axis) // 2
        return frame[y_start:y_start + shortest_axis, x_start:x_start + shortest_axis]


    def crop_bounding_box(self, frame, bb):
        frame_height, frame_width, _ = frame.shape
        resized_height, resized_width = self.resized_image_shape

        # Calculate scale ratios for x and y axes
        scale_ratio_x = frame_width / resized_width
        scale_ratio_y = frame_height / resized_height

        # Map bounding box coordinates to the high-res frame
        center_x = int((bb['x'] + bb['width'] / 2) * scale_ratio_x)
        center_y = int((bb['y'] + bb['height'] / 2) * scale_ratio_y)

        # Define target object size
        object_height, object_width = self.object_size, self.object_size

        # Calculate crop region
        x_start = center_x - (object_width // 2)
        y_start = center_y - (object_height // 2)

        # Adjust if out of bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(frame_width, x_start + object_width)
        y_end = min(frame_height, y_start + object_height)

        cropped_img = frame[y_start:y_end, x_start:x_end]
        return cropped_img


    def run_anomaly_detection(self):
        if not self.extracted_objects:
            return

        # Initialize anomaly detection runner if not already done
        if self.anomaly_detection_runner is None:
            if not os.path.isfile(self.anomaly_detection_model_file):
                print(f"Anomaly detection model file {self.anomaly_detection_model_file} does not exist.")
                return
            self.anomaly_detection_runner = ImageImpulseRunner(self.anomaly_detection_model_file)
            self.anomaly_detection_runner.init()

        for obj in self.extracted_objects:
            cropped_img = obj['cropped_image']
            anomalies = self.detect_anomalies(cropped_img)
            obj['anomalies'] = anomalies
            obj['anomaly_grid_image'] = self.draw_anomaly_grid(cropped_img, anomalies)

    def detect_anomalies(self, cropped_img):
        anomalies = []
        features, _ = self.anomaly_detection_runner.get_features_from_image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        result = self.anomaly_detection_runner.classify(features)
        if "visual_anomaly_grid" in result["result"]:
            anomalies = [{'x': grid_cell['x'], 'y': grid_cell['y'], 'width': grid_cell['width'],
                          'height': grid_cell['height'], 'confidence': grid_cell['value']} 
                          for grid_cell in result["result"]["visual_anomaly_grid"]]
        return anomalies

    def draw_anomaly_grid(self, cropped_img, anomalies):
        anomaly_grid_img = cropped_img.copy()
        for grid_cell in anomalies:
            cv2.rectangle(anomaly_grid_img, (grid_cell['x'], grid_cell['y']),
                          (grid_cell['x'] + grid_cell['width'], grid_cell['y'] + grid_cell['height']), (50, 255, 255), 2)
        return anomaly_grid_img

    def save_cropped_image_if_needed(self, cropped_img, label):
        current_time = time.time()
        if self.save_images_interval > 0 and current_time - self.last_saved_time >= self.save_images_interval:
            timestamp = int(current_time)
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, cropped_img)
            print(f"Saved cropped image: {filepath}")
            self.last_saved_time = current_time

    # Methods to get frames and data for routes
    def get_inference_frame(self):
        if self.inference_frame is not None:
            return self.inference_frame.copy()
        else:
            return None

    def get_high_res_frame(self):
        if self.latest_high_res_frame is not None:
            return self.latest_high_res_frame.copy()
        else:
            return None

    def get_bounding_boxes_with_images(self):
        enriched_bounding_boxes = []
        for obj in self.extracted_objects:
            bb = obj['bounding_box']
            cropped_img = obj['cropped_image']
            anomaly_img = obj.get('anomaly_grid_image', cropped_img)
            anomalies = obj.get('anomalies', [])

            # Encode images to base64
            cropped_image_base64 = self.encode_image_to_base64(cropped_img)
            anomaly_image_base64 = self.encode_image_to_base64(anomaly_img)

            bb_with_images = {**bb, 'cropped_image': cropped_image_base64, 'anomaly_grid_image': anomaly_image_base64, 'anomalies': anomalies}
            enriched_bounding_boxes.append(bb_with_images)
        return enriched_bounding_boxes

    def encode_image_to_base64(self, img):
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    def get_inference_speed(self):
        return self.inference_speed

    def get_object_count(self):
        return self.count_objects

# Create an instance of ImageProcessor
parser = argparse.ArgumentParser(description="Run the image processing app with camera or video file and save interval options.")
parser.add_argument('--camera', type=str, default='0', help='Camera ID or video file path to use (default: 0 for the first camera)')
parser.add_argument('--extracted-object-size', type=int, default=150, help='Size of the squared bounding boxes around the extracted objects in pixels. Increase if you have large objects, decrease if you have small objects (default: 150)')
parser.add_argument('--save-images-interval', type=int, default=0, help='Interval to save images in seconds (default: 0, meaning no saving)')
args = parser.parse_args()

processor = ImageProcessor(args.camera, args.extracted_object_size, args.save_images_interval)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/high_res_video_feed')
def high_res_video_feed():
    def gen_frames():
        while True:
            frame = processor.get_high_res_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_detection_feed')
def object_detection_feed():
    def gen_frames():
        while True:
            frame = processor.get_inference_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/extracted_objects_feed')
def extracted_objects_feed():
    enriched_bounding_boxes = processor.get_bounding_boxes_with_images()
    return jsonify(enriched_bounding_boxes)

@app.route('/inference_speed')
def inference_speed():
    def get_inference_speed():
        while True:
            speed = processor.get_inference_speed()
            yield f"data:{speed}\n\n"
            time.sleep(0.1)
    return Response(get_inference_speed(), mimetype='text/event-stream')

@app.route('/object_counter')
def object_counter():
    def get_objects():
        while True:
            count = processor.get_object_count()
            yield f"data:{count}\n\n"
            time.sleep(0.1)
    return Response(get_objects(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Set the log level to only show errors (suppress HTTP GET logs)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=5001, debug=True)
