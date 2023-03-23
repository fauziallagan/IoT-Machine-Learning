# Tulislah code untuk komputer mengenali wajah anda di bawah ini
from firebase_admin import credentials, initialize_app
from google.cloud import storage

import numpy as np
import cv2
import os
try :
    print("\n Menjalankan program... ")
    path = "Program Face Recognition (IoT)"
    credentialJson = "credentials/YOUR_CREDENTIALS_JSON" # Wajib ISI
    storageBucketToken = 'YOUR_STORAGE BUCKET_TOKEN' # Wajib ISI
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentialJson 
    cred = credentials.Certificate(credentialJson) 
    initialize_app(cred, {'storageBucket': storageBucketToken}) 

    client = storage.Client('trainingdata')
    bucket = client.get_bucket(storageBucketToken)
    blob = bucket.blob("trainer.yml")

    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    
    blob.download_to_filename("trainer/trainer.yml")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0

    names = ['zi', 'fauzi', 'NAMA_PESERTA_2'] # Wajib ISI (Nama Peserta)

    cam = cv2.VideoCapture(0) # sesuaikan dengan default kamera pada pc (0 untuk webcam internal 1 untuk webcam eksternal)
    cam.set (3, 640)
    cam.set (4, 480)

    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
except:
    print("\n Error")
    
print("\n Mulai menyalakan Kamera.... ")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

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

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff #'esc' buat keluar

    if k == 27:
        break

print("\n Keluar program...")
print("\n Program Selesai)
cam.release()
cv2.destroyAllWindows()
            
