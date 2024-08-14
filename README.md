# Face Recognition Project

This project implements a face recognition system using MTCNN (Multi-task Cascaded Convolutional Networks) for face detection and FaceNet for generating face embeddings. 

## Project Overview

### MTCNN
MTCNN is a powerful tool for detecting faces in images. It utilizes a deep learning-based approach to simultaneously detect faces and align them by detecting facial landmarks. This ensures that the faces are properly cropped and aligned for the next stage of processing.

### FaceNet
FaceNet is a deep learning model that transforms images into a compact, high-dimensional feature vector (embedding) that represents the unique features of a face. These embeddings are then used for tasks such as face recognition and clustering.

### Workflow
1. **Data Collection & Labeling**: First, gather and label the dataset of faces.
2. **Image Reading**: Use OpenCV to read the images.
3. **Face Detection**: Detect faces using MTCNN.
4. **Model Training**: Extract features from the detected faces using FaceNet, and then use SVM (Support Vector Machine) from scikit-learn to train the model.
5. **Model Saving**: Save the trained model using joblib.
6. **Prediction**: Perform predictions by comparing the face embeddings with the stored ones.

## Running the Application Locally

### Step 1: Clone the GitHub Repository
```bash
git clone https://github.com/SaiKumarSeela/face_recognition_assigment
```

### Step 2: Create a Python Virtual Environment
```bash
cd face_recognition_assigment
python -m venv face_recognition
```

### Step 3: Activate the Virtual Environment
  ```bash
  face_recognition\Scripts\activate
  ```

### Step 4: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 5: Train the SVM Model
```bash
python main.py
```

### Step 6: Run the Face Recognition Application
```bash
python app.py
```

## Demo
[![Watch the video](https://img.youtube.com/vi/b4KdZaNm7Gc/hqdefault.jpg)](https://www.youtube.com/embed/b4KdZaNm7Gc)


## Conclusion

This project demonstrates how to use deep learning techniques for face recognition, leveraging the power of MTCNN for face detection and FaceNet for creating meaningful face embeddings. The SVM model is then used for classification, ensuring robust performance in recognizing faces.
