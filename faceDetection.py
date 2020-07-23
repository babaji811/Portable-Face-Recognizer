import cv2
import numpy as np
import pickle
import os, signal

#cap = cv2.VideoCapture("http://192.168.1.25:8080/video")
cap = cv2.VideoCapture(0)
cascade_path = r'C:\Users\Aashay\faceDetection\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"C:\Users\Aashay\Pictures\images\trainer.yml")

orig_labels = {}
with open(r"C:\Users\Aashay\Pictures\images\labels.pickle", "rb") as f:
    orig_labels = pickle.load(f)
    labels = {v:k for k,v in orig_labels.items()}



while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #recognizer
        id, conf = recognizer.predict(roi_gray)
        if(conf>=55):
            print(id)
            print(labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (0, 0, 0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        color = (200, 100, 100)   #BGR
        thickness = 2
        x_end = x + w
        y_end = y + h
        cv2.rectangle(frame, (x,y), (x_end, y_end), color, thickness)   #(x,y) and (x_end, y_end) are diagonal points
    cv2.imshow('webcam', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
os.kill (os.getpid(), signal.SIGTERM)

