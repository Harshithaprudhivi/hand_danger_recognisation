# Real-Time Hand/Object Danger Zone Detection – Proof of Concept (POC)

This project is a **real-time computer vision Proof of Concept (POC)** that detects when a user’s hand (represented using a **colored object**) approaches a **virtual danger zone** on the screen and displays visual warnings based on proximity.

The system is built using **classical computer vision techniques only** and runs in real time on **CPU-only execution**, achieving the required **≥ 8 FPS** performance target.

---

## ✅ Key Features

- **Real-time object tracking using color segmentation**
- **Virtual danger-zone boundary drawn on live camera feed**
- **Dynamic distance-based state classification:**
  - **SAFE** – object is far from the boundary  
  - **WARNING** – object approaching the boundary  
  - **DANGER** – object touching / extremely close to the boundary  
- **On-screen alert:** `"DANGER DANGER"` displayed during danger state
- **Smooth tracking with centroid stabilization (no jitter)**
- **CPU-only execution (no GPU, no MediaPipe, no OpenPose)**
- **Live FPS display to verify real-time performance**

---

## Technologies Used

- **Python**
- **OpenCV (cv2)** – video capture, image processing, contour detection, drawing
- **NumPy** – array operations and image masking

**Not Used:** MediaPipe, OpenPose, PyTorch, TensorFlow, or any cloud AI APIs.

---

## Computer Vision Techniques Implemented

- Color Segmentation using **HSV color space**
- Binary Mask Creation
- Morphological Operations (Erosion & Dilation)
- **Contour Detection**
- **Centroid Tracking**
- **Exponential Smoothing for Stability**
- **Distance-Based State Machine (SAFE / WARNING / DANGER)**

---

##  How It Works

1. The webcam captures live video frames.
2. A **colored object (red marker)** is detected using HSV color segmentation.
3. The largest contour of the detected color is selected.
4. The **centroid** of the contour is tracked smoothly.
5. A **virtual rectangle** acts as the danger zone.
6. The distance between the object and the rectangle is calculated.
7. Based on distance:
   - SAFE → Green
   - WARNING → Orange
   - DANGER → Red + “DANGER DANGER” warning
8. FPS is calculated and displayed in real time.

---

## Performance

- **Target FPS:** ≥ 8 FPS (CPU-only)
- **Achieved:** Typically 12–25 FPS depending on system
- **No GPU acceleration used**

