# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:08:41 2020

@author: varunraj
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while(True):
    #capture frame by frame
    ret,frame = cap.read()
    #operations on video
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        gray = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
        
    
    
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    