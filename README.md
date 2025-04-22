# Gesture_Based_Drone_Control_System

Welcome to the **Gesture Based Drone Control System**, an experimental project by Soumya Sourav that demonstrates how drones can be controlled without traditional remotes—relying entirely on gestures and speech-based inputs. This system leverages modern Machine Learning (ML) and Computer Vision (CV) techniques to interpret human actions and commands for drone operation.

---

## 🔍 Overview

This project explores multiple approaches to gesture and voice-based control, aiming to provide flexible and intuitive alternatives to remote controllers. We implement gesture recognition using CNNs, MediaPipe, and YOLO models, along with speech recognition enhanced by LLM-powered synonym understanding via the Gemini API.

---

## 📁 Repository Structure
```
Gesture-Based-Drone-Control-System/
│
├── Dataset/ # Sample dataset used for training/classical methods
├── cnnMain.py # CNN-based gesture classification and control
├── mediapipeMain.py # Hand gesture recognition using MediaPipe (no dataset needed)
├── speechMain.py # Speech-based control using Gemini API for synonym expansion
└── yoloMain.ipynb # YOLOv11n-based gesture recognition (Ultralytics)
```


---

## 📦 Components

### 1. `Dataset/`
- Contains a sample gesture dataset.
- **Credit:** Dataset is sourced from **Ultralytics**.

### 2. `cnnMain.py`
- Uses a classical Convolutional Neural Network (CNN) to classify hand gestures.
- Based on the provided dataset.
- Outputs gesture-based control commands.

### 3. `mediapipeMain.py`
- Utilizes **Google's MediaPipe** to detect and track hand keypoints.
- Doesn't require any dataset.
- Ideal for real-time gesture tracking and control.

### 4. `speechMain.py`
- Adds voice command functionality.
- Captures spoken commands and processes them using **Gemini API**, which expands synonyms for better command understanding.
- Enhances usability with natural language input.

### 5. `yoloMain.ipynb`
- Implements **YOLOv11n**, a powerful pre-trained model from **Ultralytics**.
- Used for gesture recognition.
- No fine-tuning applied yet, but performs well for initial tests.

---

## 🚀 Getting Started

To run each module, ensure required libraries are installed:

- `tensorflow`
- `mediapipe`
- `ultralytics`
- `speechrecognition`
- `Gemini API setup (for speech understanding)`

Run each script or notebook individually based on the desired functionality.

---

## 🤝 Credits

- **Ultralytics** for the dataset and the YOLOv11n model.
- **Google MediaPipe** for hand landmark tracking.
- **Gemini API** for enhancing speech-based control using AI.

---

## 📌 Note

This is a proof-of-concept system and currently supports basic gesture/speech control logic. It is designed for experimentation and development purposes—real-world drone control should include safety protocols and hardware integrations.

