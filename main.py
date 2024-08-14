import os
import cv2 as cv
import numpy as np
from joblib import dump, load
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from utils.preprocessing import FACELOADING
from utils.emdeddings import get_embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_preprocess_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def detect_faces(img, detector):
    results = detector.detect_faces(img)
    if results:
        return results[0]['box']  # Return the bounding box of the first face detected
    return None

def extract_face(img, box):
    x, y, w, h = box
    face = img[y:y+h, x:x+w]
    face = cv.resize(face, (160,160))
    return face

def load_and_embed_faces(dataset_path):
    faceloading = FACELOADING(dataset_path)
    X, Y = faceloading.load_classes()
    embedded_faces = [get_embedding(img) for img in X]
    return np.asarray(embedded_faces), Y

def train_svm_model(X_train, Y_train):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)
    train_acc = accuracy_score(Y_train, ypreds_train)
    test_acc = accuracy_score(Y_test, ypreds_test)
    return train_acc, test_acc

def predict_and_inverse_transform(model, img, detector, encoder):
    box = detect_faces(img, detector)
    if box:
        face = extract_face(img, box)
        face_embedding = get_embedding(face)
        ypreds = model.predict([face_embedding])
        return encoder.inverse_transform(ypreds)
    return None

def main():
    # Initialize face detector
    detector = MTCNN()

    # Load and preprocess a sample image
    img = load_and_preprocess_image("./dataset/robert_downey/5.jpg")
    
    # Detect and extract the face
    box = detect_faces(img, detector)
    if box:
        img = cv.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 30)
        face = extract_face(img, box)

    # Load and embed faces
    X, Y = load_and_embed_faces("./dataset")

    # Save embedded faces
    np.savez_compressed('faces_embeddings_done_4classes.npz', X, Y)

    # Encode labels
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, random_state=17)

    # Train the SVM model
    model = train_svm_model(X_train, Y_train)

    # Evaluate the model
    train_acc, test_acc = evaluate_model(model, X_train, Y_train, X_test, Y_test)
    print("Training Accuracy:", train_acc)
    print("Testing Accuracy:", test_acc)

    # Predict and inverse transform on a test image
    test_img = load_and_preprocess_image("virat_kohli_test.jpg")
    prediction = predict_and_inverse_transform(model, test_img, detector, encoder)
    print("Prediction:", prediction)

    # Save the trained model
    dump(model, 'svm_model_160x160.joblib')

if __name__ == "__main__":
    main()
