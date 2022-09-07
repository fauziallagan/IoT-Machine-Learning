import numpy as np
import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'Aqilla', 'Rahman', 'Musyaffa', 'Z', 'W']

ds_factor = 0.6

class VideoRecognizer(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set (3, 640)
        self.cam.set (4, 480)

    def __del__(self):
        self.cam.release()

    def get_frame(self):    
        ret, img = self.cam.read()

        img = cv2.resize(img,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        minW = 0.1*self.cam.get(3)
        minH = 0.1*self.cam.get(4)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW),int(minH))
           )
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if (confidence < 100):
                id = names[id]
                confidence = "   {0}%".format(round(100 - confidence))
            else:
                id = "Unknown"
                confidence = "   {0}%".format(round(100 - confidence))

            cv2.putText(
                img,
                str(id),
                (x+5, y-5),
                font,
                1,
                (255,255,255),
                2
                )

            cv2.putText(
                img,
                str(confidence),
                (x+5, y+h-5),
                font,
                1,
                (255,255,0),
                1
                )
            break

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
