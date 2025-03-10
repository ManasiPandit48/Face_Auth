import os
import pickle
import face_recognition
from datetime import datetime

class Authenticator:
    def __init__(self, db_dir='./db', log_path='./log.txt'):
        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = log_path

    def register_user(self, name, image):
        embeddings = face_recognition.face_encodings(image)[0]
        file_path = os.path.join(self.db_dir, f'{name}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(embeddings, file)
        return "User registered successfully!"

    def recognize_user(self, image):
        embeddings_unknown = face_recognition.face_encodings(image)
        if len(embeddings_unknown) == 0:
            return 'no_persons_found'
        
        embeddings_unknown = embeddings_unknown[0]
        db_dir = sorted(os.listdir(self.db_dir))

        for file_name in db_dir:
            file_path = os.path.join(self.db_dir, file_name)
            with open(file_path, 'rb') as file:
                embeddings = pickle.load(file)
                match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
                if match:
                    return file_name[:-7]  # Removing ".pickle" from the name

        return 'unknown_person'

    def log_access(self, name, action):
        with open(self.log_path, 'a') as f:
            f.write(f'{name},{datetime.now()},{action}\n')
