from firebase_admin import credentials, initialize_app, storage
from PIL import Image

import numpy as np
import cv2
import os

path = 'dataset'

cred = credentials.Certificate("credentials/YOUR_CREDENTIAL_JSON") # Wajib ISI
initialize_app(cred, {'storageBucket':'YOUR_FIREBASE_BUCKET'}) # Wajib ISI

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassif





ier('Cascades\haarcascade_frontalface_default.xml')

def getImageAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')        #grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int (os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+h])
            ids.append(id)
            
    return faceSamples, ids

print("\n Training Dataset...")

faces,ids = getImageAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer.yml')

fileName = "trainer.yml"
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)
blob.make_public()
print("\n URL untuk Mendownload File : ", blob.public_url)
print("\n Nama Bucket : ", bucket)
print("\n Nama Blob : ", blob)

os.remove("trainer.yml")

print("\n Training Data Telah Dibentuk dan Diupload ke Firebase Storage...".format(len(np.unique(ids))))
      
                  
