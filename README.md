# Hand Gesture Recognition Using CNN and LSTM

This project detects and classifies American Sign Language (ASL) gestures using a deep learning model combining **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)**. It leverages **MediaPipe** for real-time hand tracking and **Streamlit** for an interactive UI.

---

## Features

- Real-time webcam-based gesture detection
- CNN+LSTM model for sequential gesture classification
- Supports multiple ASL signs with high accuracy
- Visual hand landmarks drawn using MediaPipe
- Easily expandable to new gestures

---

## Project Structure
hand-gestures-recognition/
│
├── app.py # Main Streamlit application
├── model/ # Saved Keras model (.keras or .h5)
├── requirements.txt # List of Python dependencies
├── .gitignore # Ignore virtual environments and system files
├── signlanguage.jpg # Reference chart of ASL gestures
├── signlanguage.svg # Vector image version of ASL chart
└── README.md # Project documentation

## Getting Started

### 1. Clone the repository

```
git clone https://github.com/MayPham2571/hand-gestures-recognition.git
cd hand-gestures-recognition
```
### 2. Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate  
```
(On Windows: venv\Scripts\activate)

### 3. Install required packages
```
pip install -r requirements.txt
```
### 4. Run the application
```
streamlit run app.py
```
