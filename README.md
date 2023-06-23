#ObjectDetection

This repository contains a Python script that utilizes the Olympe library and OpenCV to detect objects in a video stream and control a drone based on the detected objects.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV (cv2)
- Olympe
- NumPy

You can install the required Python packages using pip:

```
pip install opencv-python olympe numpy
```

## Setup

1. Connect to the drone's Wi-Fi network.

2. Update the `DRONE_IP` variable in the code to match the IP address of your drone. By default, it is set to `192.168.42.1`.

3. Download the `best.onnx` file and `coco.txt` file and place them in the same directory as the script. You can find the `best.onnx` file [here](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3/model/best.onnx) and the `coco.txt` file [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names).

## Usage

1. Run the script by executing the following command:

```
python script.py
```

2. The script will establish a connection with the drone and wait for it to take off and reach a stable hover state.

3. It will then start capturing video from the drone's camera or a specified video file.

4. The script uses the YOLOv3 object detection model to detect objects in each frame of the video. Detected objects are displayed on the screen with bounding boxes and labels.

5. The drone's movement is controlled based on the detected objects. It calculates the desired velocity for the drone based on the position of the detected objects in the frame and adjusts the drone's movements accordingly.

6. Press 'q' to stop the script and exit.

**Note:** Make sure to fly the drone in a safe environment and comply with all local regulations and laws regarding drone usage.

## License

This code is licensed under the [MIT License](LICENSE). Feel free to modify and distribute it as needed.

## Acknowledgments

- The Olympe library: https://developer.parrot.com/docs/olympe/
- YOLOv3 object detection model: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov3
