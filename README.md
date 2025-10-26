# facial-emotional-detection
Facial emotion detection is a technology that identifies human emotions from facial expressions using image processing and machine learning

1. Project Objective
To develop a facial emotion recognition system that can automatically detect human emotions such as happy, sad, angry, surprise, neutral, etc. from facial expressions using a Convolutional Neural Network (CNN) and a web camera.

2. Tools & Libraries Used
<img width="902" height="215" alt="image" src="https://github.com/user-attachments/assets/f3de39fd-9454-4f6b-a5f8-87b4e10a8c0e" />

3. Dataset Collection
The dataset used is FER2013 (Facial Expression Recognition 2013) available on Kaggle.
📦 Dataset Link:
https://www.kaggle.com/datasets/msambare/fer2013

Steps:
1. Download the dataset ZIP file from Kaggle.
2. Extract it into your project folder, for example:
     D:\facial_recognization\dataset\
3. Inside it, you will have folders like:
train/
validation/
test/ 
Each folder contains images categorized by emotion (angry, happy, sad, etc.).
4. Project Folder Structure
   facial_recognization/
│
├── dataset/
│   ├── train/
│   ├── validation/
│
├── train_model.py
├── run_demo.py
├── requirements.txt
└── emotion_recognition_model.h5
5. Model Training (train_model.py)
   Step-by-Step:
   1) Import Libraries
   2) Load and Preprocess Data
   3) Build CNN Model
   4) Compile Model
   5) Set Callbacks and Train
   6) Plot Accuracy & Loss
   7) Real-Time Detection (run_demo.py)
       Step-by-Step:
      1) Import Libraries
      2) Load Model and Define Classes
      3) Access Webcam
  8) Run the Project
      1) Train Model
      2) Run Real-Time Detection
  The webcam will open and display live emotion predictions on detected faces.
