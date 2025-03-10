'''import tensorflow as tf
import numpy as np
import os
import pickle
from datetime import datetime
from PIL import Image

class Authenticator:
    def __init__(self, model_path, db_dir='./db', log_path='./log.txt'):
        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = log_path
        self.model = tf.keras.models.load_model(model_path)
        
    def _preprocess_image(self, image):
        image = image.resize((160, 160))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)

    def _get_embeddings(self, image):
        image_array = self._preprocess_image(image)
        embeddings = self.model.predict(image_array)
        return embeddings[0]

    def register_user(self, name, image):
        embeddings = self._get_embeddings(image)
        file_path = os.path.join(self.db_dir, f'{name}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)
        return "User registered successfully!"

    def recognize_user(self, image):
        embeddings_unknown = self._get_embeddings(image)
        db_dir = sorted(os.listdir(self.db_dir))

        for file_name in db_dir:
            file_path = os.path.join(self.db_dir, file_name)
            with open(file_path, 'rb') as file:
                embeddings = pickle.load(file)
                distance = np.linalg.norm(embeddings - embeddings_unknown)
                if distance < 0.6:
                    return file_name[:-7]  

        return 'unknown_person'

    def log_access(self, name, action):
        with open(self.log_path, 'a') as f:
            f.write(f'{name},{datetime.now()},{action}\n')

authenticator = Authenticator(model_path='models/model.h5')'''
