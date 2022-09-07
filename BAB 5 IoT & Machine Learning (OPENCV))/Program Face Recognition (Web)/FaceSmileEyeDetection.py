import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades\haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Cascades\haarcascade_smile.xml')

ds_factor = 0.6

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture (0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, img = self.cap.read ()

        img = cv2.resize(img,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 5,
            minSize = (30, 30)
        )

        for (x,y,w,h) in faces:
            cv2.rectangle (img, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray [y:y+h, x:x+w]      #roi : Region of Interest
            roi_color = img [y:y+h, x:x+w]

            eyes = eyeCascade.detectMultiScale(
                roi_gray,
                scaleFactor = 1.5,
                minNeighbors = 10,
                minSize = (5, 5)
            )

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor = 1.5,
                minNeighbors = 15,
                minSize = (25, 25)
            )

            for(e_x, e_y, e_w, e_h) in eyes:
                cv2.rectangle (roi_color, (e_x,e_y), (e_x+e_w, e_y+e_h), (0,255,0), 2)

            for(s_x, s_y, s_w, s_h) in smile:
                cv2.rectangle (roi_color, (s_x,s_y), (s_x+s_w, s_y+s_h), (0,0,255), 2)
                break

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    
