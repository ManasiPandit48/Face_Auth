import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Layer

# Define your custom layer
class CustomScaleLayer(Layer):
    def __init__(self, scale_factor=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

    def get_config(self):
        config = super().get_config()
        config.update({'scale_factor': self.scale_factor})
        return config

# Register the custom layer using CustomObjectScope
with CustomObjectScope({'CustomScaleLayer': CustomScaleLayer}):
    model = tf.keras.models.load_model('models/model.h5', custom_objects={'CustomScaleLayer': CustomScaleLayer})

# Rest of your code...

# Paths for storing face embeddings and labels
embeddings_path = 'embeddings.npy'
labels_path = 'labels.npy'

# Directory to store captured images
captured_images_dir = "captured_images"
os.makedirs(captured_images_dir, exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to capture face embedding
def get_face_embedding(face_image):
    face_resized = cv2.resize(face_image, (160, 160))
    face_normalized = face_resized / 255.0
    face_reshaped = face_normalized.reshape(1, 160, 160, 3)
    embedding = model.predict(face_reshaped)
    return embedding

# Register a new user
def register_user(username):
    st.write("Capturing face for registration...")
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            # Save the captured image
            image_path = os.path.join(captured_images_dir, f"{username}.jpg")
            cv2.imwrite(image_path, face)

            # Get face embeddings
            embedding = get_face_embedding(face)

            # Load or initialize embeddings and labels
            try:
                embeddings = np.load(embeddings_path)
                labels = np.load(labels_path)
            except FileNotFoundError:
                embeddings = np.array([])
                labels = np.array([])

            # Append new data
            if embeddings.size == 0:
                embeddings = embedding
            else:
                embeddings = np.vstack((embeddings, embedding))
            labels = np.append(labels, [username])

            # Save updated data
            np.save(embeddings_path, embeddings)
            np.save(labels_path, labels)

            st.write(f"User {username} registered successfully! Image saved at {image_path}")
            break

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# Login user
def login_user():
    st.write("Capturing face for login...")
    cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            # Get face embeddings
            embedding = get_face_embedding(face)

            try:
                embeddings = np.load(embeddings_path)
                labels = np.load(labels_path)

                # Compute cosine similarity with stored embeddings
                similarities = cosine_similarity(embedding, embeddings)
                index = np.argmax(similarities)

                if similarities[0][index] > 0.8:  # Threshold for a match
                    st.write(f"Welcome, {labels[index]}!")
                    break
                else:
                    st.write("Authentication failed. Please try again.")
            except FileNotFoundError:
                st.write("No registered users found. Please register first.")
                break

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# Streamlit app layout
st.title("Real-Time Face Authentication System")

choice = st.sidebar.selectbox("Choose an option", ["Login", "Register"])

if choice == "Register":
    username = st.text_input("Enter username for registration:")
    if st.button("Register"):
        register_user(username)

elif choice == "Login":
    if st.button("Login"):
        login_user()
