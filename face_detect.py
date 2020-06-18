# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:02:35 2020

@author: varunraj
"""
#importing necessary libraries
import cv2
import argparse
import numpy as np
import time

#importing weights and config file
def load_yolo():
    net = cv2.dnn.readNetFromDarknet("yolov3-face.cfg","yolov3-wider_16000.weights")
    classes = []
    with open("face.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_name = net.getLayerNames()
    output_layers = [layers_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(img,net,outputlayers):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416,416),
                                     [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputlayers)
    return blob, outputs

def get_box_dimensions(outputs,height,width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes,confs,class_ids


def draw_labels(boxes, confs, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("img",img)
    
    

def face_detect():
    model, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        height,width,channels = frame.shape
        blob, outputs = detect_objects(frame,model,output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs,height,width)
        draw_labels(boxes,confs,class_ids,classes,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
          
                
          
#run this function to detect faces
#face_detect()



# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()                
            
                