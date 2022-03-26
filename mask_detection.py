# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:06:23 2021

@author: Beyza
"""

import cv2
import numpy as np

img = cv2.imread("C:/YOLO/mask detection yolo/mask3.jpg")

img_with = img.shape[1]
img_hight = img.shape[0]

img_blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)

labels = ["Risky","Risk free"]



colors = ["0,255,255","255,0,0"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(18,1))


model = cv2.dnn.readNetFromDarknet("C:/YOLO/mask detection yolo/yolov3_mask.cfg","C:/YOLO/mask detection yolo/yolov3_mask_last.weights")

layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)

#### NON-MAXIMUM SUPPRESSION

ids_list = []
boxes_list =[]
confidences_list=[]


for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id =np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.30:
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([img_with,img_hight,img_with,img_hight])
            (box_center_x,box_center_y , box_with, box_hight) = bounding_box.astype("int")
            
            star_x = int(box_center_x - (box_with/2))
            star_y = int(box_center_y - (box_hight/2))
            
            #### NON-MAXIMUM SUPPRESSION
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([star_x,star_y,int(box_with), int(box_hight)])
            
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

for max_id in max_ids:
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]
    
    star_x=box[0]
    star_y=box[1]
    box_with=box[2]
    box_hight=box[3]
    
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]
    
    
    end_x = star_x + box_with
    end_y = star_y + box_hight
            
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
            
    cv2.rectangle(img,(star_x,star_y), (end_x,end_y), box_color,1)
    cv2.putText(img,label,(star_x,star_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

        

cv2.imshow("Detection window", img)