# Object Detection System

A desktop application for real-time object detection using a custom YOLO model.
Built with Python, OpenCV, Tkinter, and Ultralytics YOLO.

---

## Features

* Detect objects in images
* Real-time detection using webcam
* Adjustable confidence threshold
* Simple GUI interface
* Uses a custom trained YOLO model

---

## Tech Stack

* Python 3.12
* OpenCV (`cv2`)
* Tkinter (GUI)
* Pillow (image handling)
* Ultralytics YOLOv8

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ObjectDetectionSystem.git
cd Objectdetectionsystem
```

Install dependencies:

```bash
pip install opencv-python pillow ultralytics
```

---

## How to Run

```bash
python Detect.py
```

---

## Usage

* **Image Mode**

  * Click "Select Image"
  * Choose an image file
  * The app will detect and display objects

* **Camera Mode**

  * Click "Start Camera"
  * Live detection will begin
  * Click "Stop Camera" to end

* Adjust confidence using the slider


## Notes

* Make sure to choose one of these models `DrugsModel.pt`,`my_custom_model.pt`,`yolo11s.pt` and put it in the same directory as `Detect.py`
* Webcam is required for camera mode
* Performance depends on your GPU/CPU

---

## Future Improvements

* Save detection results (images/video)
* Add model selection
* Improve UI design
* Add FPS counter

---

## Author

* Your Name
