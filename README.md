# 🚨 CollisionVision

A real-time computer vision system for detecting, tracking, and estimating potential collisions between objects — such as phones and bottles (configurable) — using YOLOv8 and ByteTrack.

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

git clone https://github.com/sm-abbas19/CollisionVision.git
cd CollisionVision

```

2. Set up the environment

   
Using Conda (recommended):
```bash
conda env create -f environment.yml
conda activate CollisionVision
```

### Additional Requirements

- Windows users: Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

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

- FRAME_SKIP: Run detection every N frames 
- TRAIL_LENGTH: Number of points in the motion trail 
- track_thresh: Minimum confidence for tracking 
- match_thresh: IoU threshold for matching detections to tracks 
- track_buffer: Frames to keep lost tracks 
- frame_rate: Frames per second from your camera
- Objects filtered for detection
  
In tracker/kalmanfilter.py:
- std_weight_position: Standard deviation weight for position (affects Kalman filter smoothness)
- std_weight_velocity: Standard deviation weight for velocity


In logic/collision.py, you can tune collision sensitivity:

- threshold: Distance at which collision probability is 0.5 (lower = more sensitive)
- scale: Controls how quickly probability rises as distance decreases

In the helpers/ directory, you’ll find modified ByteTrack files. Replace the default ByteTrack files in your environment with these to ensure proper functionality.

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
- This is a learning project — my first time working with computer vision. Many components and parameters (e.g., Kalman filter, collision thresholds) likely require further tuning for real-world reliability and robustness.
