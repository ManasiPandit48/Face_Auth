import streamlit as st
from PIL import Image
import cv2
import numpy as np
from authentication import Authenticator

authenticator = Authenticator()

def capture_image():
    img_file_buffer = st.camera_input("Capture Image")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        return np.array(image)
    return None

def main():
    st.title("Face Authentication System")
    
    menu = ["Login", "Register New User"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        st.subheader("Login")
        image = capture_image()
        if image is not None:
            if st.button("Login"):
                name = authenticator.recognize_user(image)
                if name in ['unknown_person', 'no_persons_found']:
                    st.error("Unknown user. Please register new user or try again.")
                else:
                    st.success(f"Welcome back, {name}!")
                    authenticator.log_access(name, 'in')

    elif choice == "Register New User":
        st.subheader("Register New User")
        name = st.text_input("Enter your name")
        image = capture_image()
        if image is not None and name:
            if st.button("Register"):
                message = authenticator.register_user(name, image)
                st.success(message)

if __name__ == "__main__":
    main()
