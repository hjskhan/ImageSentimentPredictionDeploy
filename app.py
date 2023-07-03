import tensorflow as tf
from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import os
import tempfile

app = Flask(__name__)

model = load_model('models/model_4.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def detect_emotion():
    if request.method == 'POST':
        image_upload = request.files['image']
        # Save the uploaded image to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, image_upload.filename)
        image_upload.save(temp_path)
        # Read the image using OpenCV
        img = cv2.imread(temp_path)
        # Detect faces using RetinaFace
        img_faces = RetinaFace.detect_faces(img)
        # Check if any faces are detected
        if len(img_faces) > 0 and isinstance(img_faces, dict):
            # Choose the first face detected
            face_index = list(img_faces.keys())[0]
            facial_parts = img_faces[face_index]
            recognize_face_area = facial_parts["facial_area"]
            face_img = img[recognize_face_area[1]:recognize_face_area[3], recognize_face_area[0]:recognize_face_area[2]]
            img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            # Resize and normalize the face image
            img_resized = cv2.resize(img_gray, (48, 48))
            img_normalized = img_resized / 255.0
            img_final = np.expand_dims(img_normalized, axis=0)

            # Make predictions
            predictions = model.predict(img_final)
            # Thresholding
            threshold = 0.5
            # Determine the predicted emotion class and accuracy
            if predictions[0] > threshold:
                predicted_class = 'Sad Person'
                accuracy = predictions[0][0]
            else:
                predicted_class = 'Happy Person'
                accuracy = 1 - predictions[0][0]
            os.remove(temp_path)
            return render_template('result.html', result=predicted_class, accuracy=accuracy)
        else:
            os.remove(temp_path)
            return render_template('result.html', result=None, accuracy=0.0)

if __name__ == '__main__':
    app.run(debug=True)
    