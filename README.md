# Multiple Impulses with Python: Object Detection and Anomaly Detection

This Flask-based web application demonstrates the use of **multiple impulses** for real-time object detection **and** visual anomaly detection.

The first stage detects objects using Edge Impulse’s **FOMO (Fast Object Detection)** model, then maps the detected objects back onto the high-resolution input image, extracts (crops) the objects, and finally applies **FOMO-AD**, a **visual anomaly detection** model to the cropped objects.

![Web app overview](/templates/assets/web-app-overview.png)

## Application Workflow

1. **Video process**:
   Start high-res frame capture/reading in a background thread using OpenCV.

2. **Object Detection (FOMO)**:
  The FOMO model detects objects in the frame and returns their bounding boxes. The center of each detected object is calculated.

1. **Mapping and Cropping**:
  The application maps each detected object's center back to the original high-resolution image and crops out the region around the object.

1. **Anomaly Detection**:
  Each cropped object is passed through a visual anomaly detection model, and the results are displayed, including any detected anomalies.

1. **Saving Cropped Images (Optional)**:
  Cropped objects can be saved at a user-specified interval. The images are saved with both the original crop and the overlaid anomaly grid. These images can then be imported in another Edge Impulse project to train your anomaly detection model.

## Prerequisites

Ensure you have Python 3.7 or above installed along with the following packages:

### Required Python Packages:


- `opencv-python`
- `Flask`
- `edge-impulse-linux`

### Installation Instructions

1. Clone this repository or download the source code.

2. Install the required dependencies using `pip`:

```bash
pip install opencv-python flask edge-impulse-linux
```

3. Ensure the required models (`modelfile-fomo.eim` for object detection and `modelfile-fomoad.eim` for visual anomaly detection) are placed in the same directory as the script. These models should be exported from Edge Impulse.

You can clone the following projects used to write this tutorial:

FOMO: [Cubes on a conveyor belt v3 - Labeled - 4 colors](https://studio.edgeimpulse.com/public/494157/latest)
FOMO-AD: [Cubes - Visual AD](https://studio.edgeimpulse.com/public/517331/latest)

To download the models, you can either use the "Deployment" option from Edge Impulse Studio or use the [Edge Impulse Linux CLI](https://docs.edgeimpulse.com/docs/tools/edge-impulse-for-linux/linux-node-js-sdk)

```bash
edge-impulse-linux-runner --download modelfile.eim
```

## Running the Application

To run the application, use the following command. You can specify either a camera ID or a video file path as the input:

```
python app.py --camera 0 --save-images-interval 10
```

- **`--camera`**: The camera ID (default: `0` for the first camera) or a video file path.
- **`--save-images-interval`**: (Optional) The interval in seconds for saving processed images (default: `0`, meaning no images will be saved).

For example, to run the app using a video file and save images every 10 seconds, use:

```bash
python app.py --camera 0
```
or
```bash
python app.py --camera /path/to/video.mp4 --save-images-interval 10
```

*In the `input/` folder, you can find three videos for testing purposes.*

Go to [http://localhost:5001/](http://localhost:5001/).

![Overview](/templates/assets/multi-impulse-linux-overview.gif)

## Available Routes

1. **`/`**: The main page that displays high-resolution and processed inference video feeds.

2. **`/high_res_video_feed`**: Provides the high-resolution video feed (from the camera or video file).

3. **`/video_feed`**: Provides the video feed with object detection, centroids, and anomaly detection results.

4. **`/inference_speed`**: Returns the inference speed (time taken for the FOMO detection process) as an event stream. - TODO: Capture the full process speed.

5. **`/object_counter`**: Returns the number of detected objects as an event stream.

6. **`/bounding_boxes_feed`**: Provides bounding boxes, cropped images, and anomaly results in JSON format.

## Notes

- The models must be located in the same directory as the script before running the application.
- This application is a demonstration and is not optimized for production.
- If you are planning on using this on a distant device and don't have access to a monitor connected to that device, replace the last line with: app.run(host="0.0.0.0", port=5001, debug=True), it will broadcast the web application to your local network.
