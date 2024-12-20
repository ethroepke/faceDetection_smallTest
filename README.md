# FaceDetection-AlertSystem

This project implements a real-time face detection system using OpenCV. It identifies faces from a pre-saved dataset, plays an audio alert upon detection, and sends an SMS notification through Twilio. Ideal for small-scale security setups or personalized automation projects.

---

## Features
- Real-time face detection using OpenCV.
- Face matching against a dataset of saved images.
- Text-to-speech audio alert for identified faces.
- SMS notification using Twilio when a face is recognized.

---

## Prerequisites
Ensure the following are installed on your system:
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- pyttsx3 (Text-to-speech library)
- Twilio Python SDK (`twilio`)

Install the required Python packages:
```bash
pip install opencv-python numpy pyttsx3 twilio
