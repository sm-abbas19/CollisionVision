# 🚨 Collision Estimator

A real-time computer vision system for detecting, tracking, and estimating potential collisions between objects (phones and bottles (can be adjusted)) using **YOLOv8** and **ByteTrack**.

---

## 🔧 Features

- 🧠 **Real-time object detection** using YOLOv8  
- 🎯 **Multi-object tracking** via ByteTrack  
- ⚠️ **Collision probability estimation** based on object distance and motion  
- 🌀 **Motion trail visualization** for tracked objects  
- ⚙️ **Customizable detection, tracking, and collision parameters**

---



---

## 📦 Installation

### 1. Clone the repository

```bash

git clone https://github.com/yourusername/Collision_Estimator.git
cd Collision_Estimator

```

2. Set up the environment

   
Using Conda (recommended):
```bash
conda env create -f environment.yml
conda activate collision-detector
```

Or install dependencies manually:

```bash
pip install numpy opencv-python matplotlib ultralytics cython lap filterpy scikit-learn
```

## 🚀 Usage
Run the application:
```bash
python main.py
```
Press ESC to exit the application.


## ⚙️ How It Works

- Detection: YOLOv8 detects phones (class 67) and bottles (class 39) in each frame.
- Tracking: ByteTrack tracks object identities across frames — even during occlusion.
- Collision Estimation: For the two closest objects, the system estimates if they’re approaching and computes a collision probability using a sigmoid function based on their distance.
- Visualization: The video stream shows bounding boxes, object IDs, motion trails, and Collision probabilities.



## 🛠️ Configuration

You can adjust key parameters in main.py:

- FRAME_SKIP: Run detection every N frames (default: 1)
- TRAIL_LENGTH: Number of points in the motion trail (default: 20)
- track_thresh: Minimum confidence for tracking (default: 0.3)
- match_thresh: IoU threshold for matching detections to tracks (default: 0.8)
- track_buffer: Frames to keep lost tracks (default: 5)
- Objects filtered for detection

In logic/collision.py, you can tune collision sensitivity:

- threshold: Distance at which collision probability is 0.5 (lower = more sensitive)
- scale: Controls how quickly probability rises as distance decreases

## 🗂️ Project Structure
```
Collision_Estimator/
├── detector/
│   └── yolov8_wrapper.py
├── tracker/
│   └── bytetrack_wrapper.py
├── logic/
│   └── collision.py
├── utils/
│   └── visualizer.py
├── main.py
├── environment.yml
└── README.md
```


## ⚠️ Limitations

- Collision estimation is 2D (image plane only)
- Performance depends on video quality and lighting conditions
