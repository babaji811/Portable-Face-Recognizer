import os
import cv2
import numpy as np
import signal
from PIL import Image
import pickle


os.chdir(r"C:\Users\Aashay\Pictures\images")
image_dir = os.getcwd()

face_cascade = cv2.CascadeClassifier(r'C:\Users\Aashay\faceDetection\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}  #key-label/name : value-id
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if(file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg') or file.endswith('JPG')):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace('-', ' ').lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id = label_ids[label]
            pil_img = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_img.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)

with open(r"C:\Users\Aashay\Pictures\images\labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save(r"C:\Users\Aashay\Pictures\images\trainer.yml")

os.kill(os.getpid(), signal.SIGTERM)
