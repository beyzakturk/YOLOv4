# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:11:07 2021

@author: Beyza
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    
    ret ,frame = cap.read()
    frame_width = frame.shape[1]
    frame_hight = frame.shape[0]
    
    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)
    
    labels = ["kisi","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
              "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
              "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
              "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
              "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
              "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
              "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
              "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
              "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    
    
    
    colors = ["0,255,255","255,0,0","255,0,255","0,0,255","0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))


    model = cv2.dnn.readNetFromDarknet("C:/YOLO/pretrained_model/yolov3.cfg","C:/YOLO/pretrained_model/yolov3.weights")
    
    layers = model.getLayerNames()
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)
    
   
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id =np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.30:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_hight,frame_width,frame_hight])
                (box_center_x,box_center_y , box_with, box_hight) = bounding_box.astype("int")
                
                star_x = int(box_center_x - (box_with/2))
                star_y = int(box_center_y - (box_hight/2))
                
                
                end_x = star_x + box_with
                end_y = star_y + box_hight
                            
                box_color = colors[predicted_id]
                box_color = [int(each) for each in box_color]
                            
                cv2.rectangle(frame,(star_x,star_y), (end_x,end_y), box_color,1)
                cv2.putText(frame,label,(star_x,star_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
                
                
                
                        
            
    cv2.imshow("Detection window", frame)
            
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
   

cap.release()
cv2.destroyAllWindows()
