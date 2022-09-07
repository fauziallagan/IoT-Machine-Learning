import numpy as np
import cv2

cap = cv2.VideoCapture(0)       #function yang digunakan untuk mengambil file video/gambar dari camera
                                #angka 0 berarti default camera


cap.set(3,640)      #set lebar
cap.set(4,480)      #set tinggi

while (True):
    ret, frame = cap.read()     #variable "frame" yang berfungsi membaca file video/gambar
                                #ketika sebuah frame terbaca, maka function akan me-return nilai TRUE
                                #ketika tidak ada frame yang terbaca, maka function akan me-return nilai FALSE
                                #FALSE berarti camera diputuskan / tidak ada frame lagi di file video/gambar
                                #kalau nilai TRUE, maka program dilanjutkan
                                #kalau nilai FALSE, maka program berhenti

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #variable "gray" yang berfungsi mengubah file RGB yang terbaca "frame" menjadi gambar GRAY

    cv2.imshow('frame', frame)      #function yang akan menampilkan video feed
    cv2.imshow('gray', gray)        #' ... ', -menjadi nama windows dari video feed
                                    # , ...   -variable yang ingin ditampilkan pada windows

    k = cv2.waitKey(30) & 0xff

    if k == 27:     #tekan 'esc' (27) untuk keluar
        break

cap.release()       #function yang akan mematikan camera

cv2.destroyAllWindows()     #function yang akan menutup semua windows
