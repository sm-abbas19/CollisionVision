# 🚨 Collision Estimator

A real-time computer vision system for detecting, tracking, and estimating potential collisions between objects (phones and bottles) using **YOLOv8** and **ByteTrack**.

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

bash

git clone https://github.com/yourusername/Collision_Estimator.git
cd Collision_Estimator

2. Set up the environment
Using Conda (recommended):
bashCopyEditconda env create -f environment.yml
conda activate collision-detector

Or install dependencies manually:
bashCopyEditpip install numpy opencv-python matplotlib ultralytics cython lap filterpy scikit-learn


## 🚀 Usage
Run the application:
bashCopyEditpython main.py


Press ESC to exit the application.


##⚙️ How It Works


Detection: YOLOv8 detects phones (class 67) and bottles (class 39) in each frame.


Tracking: ByteTrack tracks object identities across frames — even during occlusion.


Collision Estimation: For the two closest objects, the system estimates if they’re approaching and computes a collision probability using a sigmoid function based on their distance.


Visualization: The video stream shows bounding boxes, object IDs, motion trails, and Collision probabilities.



##🛠️ Configuration
You can tweak the following settings in main.py:
ParameterDescriptionDefaultFRAME_SKIPRun detection every N frames1TRAIL_LENGTHNumber of points in the motion trail20track_threshMinimum detection confidence for tracking0.3match_threshIoU threshold to match detections to tracks0.8track_bufferHow long to retain lost tracks (in frames)5
In logic/collision.py, you can adjust collision sensitivity:
ParameterDescriptionthresholdDistance at which probability = 0.5 (lower = more sensitive)scaleControls how quickly probability increases

##🗂️ Project Structure
cssCopyEditCollision_Estimator/
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


##⚠️ Limitations


Collision estimation is 2D (image plane only)


Performance depends on video quality and lighting conditions
