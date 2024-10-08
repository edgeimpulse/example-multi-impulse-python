import argparse
import cv2
import os
import time
import json
import logging
import base64
from flask import Flask, render_template, Response, jsonify
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__, static_folder='templates/assets')

# Global variables

camera_id = 0
countObjects = 0
inferenceSpeed = 0
od_model_parameters = 0
save_images_interval = 0  # Interval in seconds to save images (0 means no saving)
last_saved_time = 0  # Keeps track of when the last image was saved

scaleFactor = 6  # Scale factor for resizing inference frames
bounding_boxes = []
latest_high_res_frame = None

desired_fps = 30
object_size = 0

# Ensure the output directory exists
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility function to get current time in milliseconds
def now():
    return round(time.time() * 1000)

# Function to save cropped images
def save_cropped_image(cropped_img, label, timestamp):
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, cropped_img)
    print(f"Saved cropped image: {filepath}")

# High-resolution video feed (from camera or looping video file)
def gen_high_res_frames():
    global latest_high_res_frame
    
    # Check if `camera_id` is a valid video file path or a camera ID
    if os.path.isfile(camera_id):
        print(f"Opening video file: {camera_id}")
        camera = cv2.VideoCapture(camera_id)
        is_video_file = True
    else:
        print(f"Using camera ID: {camera_id}")
        camera = cv2.VideoCapture(int(camera_id))  # Treat as camera ID
        is_video_file = False

    delay_between_frames = 1 / desired_fps  # Approx. 0.033 seconds for 30 FPS

    while True:
        start_time = time.time()
        success, frame = camera.read()
        
        if not success and is_video_file:
            print("End of video file, looping back to start.")
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
            continue
        elif not success:
            print("Could not get frame (camera failure)")
            break
        
        # Store the latest frame globally
        latest_high_res_frame = frame.copy()
        
        if latest_high_res_frame is not None:
            # Convert the latest frame to JPEG
            ret, buffer = cv2.imencode('.jpg', latest_high_res_frame)
            frame = buffer.tobytes()

            # Yield the frame as a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
        # Calculate time to sleep to maintain 30 FPS
        frame_process_time = time.time() - start_time
        time_to_sleep = max(0, delay_between_frames - frame_process_time)
        time.sleep(time_to_sleep)
    
    camera.release()

# Function to generate frames and run inference
def gen_frames():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, 'modelfile-fomo.eim')
    global countObjects, bounding_boxes, inferenceSpeed, latest_high_res_frame, od_model_parameters

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            # print(json.dumps(model_info, indent=4))
            od_model_parameters = model_info['model_parameters']
            while True:
                if latest_high_res_frame is None:
                    print("Waiting for high-res frame...")
                    time.sleep(1)
                    continue

                img = cv2.cvtColor(latest_high_res_frame.copy(), cv2.COLOR_BGR2RGB)
                features, cropped = runner.get_features_from_image(img)
                res = runner.classify(features)

                if "result" in res:
                    cropped = cv2.resize(cropped, (cropped.shape[1] * scaleFactor, cropped.shape[0] * scaleFactor))
                    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    cropped = process_inference_result(res, cropped)

                ret, buffer = cv2.imencode('.jpg', cropped)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            runner.stop()

# Helper function to process inference results and return the updated frame
def process_inference_result(res, cropped):
    global countObjects, bounding_boxes, inferenceSpeed

    countObjects = 0
    bounding_boxes.clear()
    inferenceSpeed = res['timing']['classification']
    if "bounding_boxes" in res["result"]:
        for bb in res["result"]["bounding_boxes"]:
            if bb['value'] > 0:
                countObjects += 1
                bounding_boxes.append({
                    'label': bb['label'],
                    'x': int(bb['x']),
                    'y': int(bb['y']),
                    'width': int(bb['width']),
                    'height': int(bb['height']),
                    'confidence': bb['value']
                })
                cropped = draw_centroids(cropped, bb)  # Update cropped frame with bounding box
    return cropped  # Return the updated frame

# Helper function to draw bounding box and centroid, and return the modified image
def draw_centroids(cropped, bb):
    center_x = int((bb['x'] + bb['width'] / 2) * scaleFactor)
    center_y = int((bb['y'] + bb['height'] / 2) * scaleFactor)
    cropped = cv2.circle(cropped, (center_x, center_y), 10, (0, 255, 0), 2)
    label_text = f"{bb['label']}: {bb['value']:.2f}"
    label_position = (int(bb['x'] * scaleFactor), int(bb['y'] * scaleFactor) - 10)
    cv2.putText(cropped, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cropped  # Return the modified frame

# Get cropped image with anomaly detection as base64
def get_cropped_image_base64_with_anomalies(box):
    global latest_high_res_frame, object_size

    height, width, _ = latest_high_res_frame.shape
    squared_frame = get_squared_image_from_high_res_frame(latest_high_res_frame, height, width)
    cropped_img = crop_bounding_box(squared_frame, box, object_size)
    cropped_img = cv2.resize(cropped_img, (object_size, object_size))
    anomalies = detect_anomalies(cropped_img)
    cropped_with_anomaly_grid = draw_anomaly_grid(cropped_img, anomalies)

    return encode_images_to_base64(cropped_img, cropped_with_anomaly_grid), anomalies, cropped_img

# Crop a square frame from the center
def get_squared_image_from_high_res_frame(frame, height, width):
    height, width, _ = latest_high_res_frame.shape
    shortest_axis = min(height, width)
    x_start = (width - shortest_axis) // 2
    y_start = (height - shortest_axis) // 2
    return frame[y_start:y_start + shortest_axis, x_start:x_start + shortest_axis]

# Crop the bounding box from the square-cropped frame
def crop_bounding_box(frame, box, object_size):
    global od_model_parameters
    frame_height, frame_width, _ = frame.shape

    # Calculate scale ratio to map bounding box to the frame
    scale_ratio = min(frame_height, frame_width) / od_model_parameters['image_input_height']
    center_x = int((box['x'] + box['width'] / 2) * scale_ratio)
    center_y = int((box['y'] + box['height'] / 2) * scale_ratio)

    # Define the target object size (fixed size)
    object_height, object_width = object_size, object_size

    # Calculate the top-left corner of the crop region
    x_start = center_x - (object_width // 2)
    y_start = center_y - (object_height // 2)

    # Ensure the crop stays within the frame boundaries by adjusting the crop region if needed
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0

    # Calculate the bottom-right corner of the crop region
    x_end = x_start + object_width
    y_end = y_start + object_height

    # Adjust the crop region if the bottom-right corner goes outside the frame
    if x_end > frame_width:
        x_start = frame_width - object_width
        x_end = frame_width
    if y_end > frame_height:
        y_start = frame_height - object_height
        y_end = frame_height

    # Ensure that the crop region is valid (i.e., not negative)
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(frame_width, x_end)
    y_end = min(frame_height, y_end)

    # Crop the frame to the fixed size
    cropped_img = frame[y_start:y_end, x_start:x_end]
    # print(f"Cropping at: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}, cropped_img.shape={cropped_img.shape}")

    return cropped_img


# Encode images to base64
def encode_images_to_base64(cropped_img, anomaly_img):
    _, buffer_original = cv2.imencode('.jpg', cropped_img)
    _, buffer_with_grid = cv2.imencode('.jpg', anomaly_img)
    return base64.b64encode(buffer_original).decode('utf-8'), base64.b64encode(buffer_with_grid).decode('utf-8')

# Detect anomalies using a second model
def detect_anomalies(cropped_img):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, 'modelfile-fomoad.eim')
    anomalies = []

    # Check if the model file exists
    if not os.path.isfile(modelfile):
        # print(f"Model file {modelfile} does not exist. Returning empty anomalies.")
        return anomalies
    
    with ImageImpulseRunner(modelfile) as anomaly_runner:
        try:
            anomaly_runner.init()
            features, _ = anomaly_runner.get_features_from_image_auto_studio_setings(cropped_img)
            result = anomaly_runner.classify(features)
            if "visual_anomaly_grid" in result["result"]:
                anomalies = [{'x': grid_cell['x'], 'y': grid_cell['y'], 'width': grid_cell['width'],
                            'height': grid_cell['height'], 'confidence': grid_cell['value']} 
                            for grid_cell in result["result"]["visual_anomaly_grid"]]
        finally:
            anomaly_runner.stop()
    return anomalies

# Draw anomaly grid on the image
def draw_anomaly_grid(cropped_img, anomalies):
    anomaly_grid_img = cropped_img.copy()
    for grid_cell in anomalies:
        cv2.rectangle(anomaly_grid_img, (grid_cell['x'], grid_cell['y']),
                      (grid_cell['x'] + grid_cell['width'], grid_cell['y'] + grid_cell['height']), (50, 255, 255), 2)
    return anomaly_grid_img


# API to get bounding boxes with cropped images and anomaly grids
@app.route('/extracted_objects_feed')
def extracted_objects_feed():
    global last_saved_time
    enriched_bounding_boxes = []
    current_time = time.time()

    save_images = save_images_interval > 0 and current_time - last_saved_time >= save_images_interval
    timestamp = int(current_time) if save_images else None

    for box in bounding_boxes:
        cropped_image, anomalies, cropped_img_resized = get_cropped_image_base64_with_anomalies(box)
        box_with_image = {**box, 'cropped_image': cropped_image[0], 'anomaly_grid_image': cropped_image[1], 'anomalies': anomalies}
        enriched_bounding_boxes.append(box_with_image)

        if save_images:
            save_cropped_image(cropped_img_resized, box['label'], timestamp)

    if save_images:
        last_saved_time = current_time

    return jsonify(enriched_bounding_boxes)

# Main route to serve the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/high_res_video_feed')
def high_res_video_feed():
    return Response(gen_high_res_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routes for live video feeds
@app.route('/object_detection_feed')
def object_detection_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Data sources for inference speed and object counting
@app.route('/inference_speed')
def inference_speed():
    def get_inference_speed():
        while True:
            yield f"data:{inferenceSpeed}\n\n"
            time.sleep(0.1)
    return Response(get_inference_speed(), mimetype='text/event-stream')

@app.route('/object_counter')
def object_counter():
    def get_objects():
        while True:
            yield f"data:{countObjects}\n\n"
            time.sleep(0.1)
    return Response(get_objects(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the image processing app with camera or video file and save interval options.")
    parser.add_argument('--camera', type=str, default='0', help='Camera ID or video file path to use (default: 0 for the first camera)')
    parser.add_argument('--extracted-object-size', type=int, default=150, help='Size of the squared bounding boxes around the extracted objects in pixels. Increase if you have large objects, decrease if you have small objects (default: 150)')
    parser.add_argument('--save-images-interval', type=int, default=0, help='Interval to save images in seconds (default: 0, meaning no saving)')
    args = parser.parse_args()

    # Set the global variables for camera or video file and save interval
    camera_id = args.camera
    object_size = args.extracted_object_size
    save_images_interval = args.save_images_interval

    print(f"Running with camera/video: {camera_id} and save images interval of {save_images_interval} seconds.")
    
    # Run the Flask app
    # Set the log level to only show errors (suppress HTTP GET logs)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=5001, debug=True)
