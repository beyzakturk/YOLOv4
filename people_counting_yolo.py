# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:03:00 2021

@author: Beyza
"""

import cv2
import numpy as np

peopleout,peoplein = 0,0
line = 300 

cap = cv2.VideoCapture('peopleCount.mp4')

contours_previous = []
contours_now = []

while True:
    
    ret ,frame = cap.read()
    frame_width = frame.shape[1]
    frame_hight = frame.shape[0]
    
    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False) #bir format türüdür 
    
    labels = ["person"] #tanıması gereken labellar
    
    
    
    colors = ["0,255,255"] #kendine özgü rengini belirttim
    colors = [np.array(color.split(",")).astype("int") for color in colors] #tüm elemanları gezerek int formatına dönüştürerek atama yaptım
    colors = np.array(colors) #tek array içinde bulunsun dedim
    colors = np.tile(colors,(18,1)) # colors matrisindeki sayıları arttırmak/büyütmak içim bunu hazır fonksiyon tile kullandım.

#modelimizi kodumuza dahil edelim
    model = cv2.dnn.readNetFromDarknet("C:/YOLO/pretrained_model/yolov3.cfg","C:/YOLO/pretrained_model/yolov3.weights")
#bulma yani detection işlemini yapabilmek için modeldeki layerları çekmem gerek    
    layers = model.getLayerNames()
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()] #outputlar gerekliydi sadece ve 1 eksiği olduğunu variabla explorer kısmından da gözlemleyebiliriz
#modele blop formatını veriyorum   
    model.setInput(frame_blob)
 #çıktıları forward metodu kullanarak değerleri elde edelim   
    detection_layers = model.forward(output_layer)
    
   
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id =np.argmax(scores) #en büyüğünü çektim
            confidence = scores[predicted_id] #güven skorunu tutmak istedim
            
            if confidence > 0.30: #box çizilip çizilmemesini belirledik
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_hight,frame_width,frame_hight]) #değerlerin anlamlı olması için koordinatlarla çarptım
                (box_center_x,box_center_y , box_with, box_hight) = bounding_box.astype("int") #aldığım değeri int tipine çevirdim
                
                star_x = int(box_center_x - (box_with/2)) #başlangıç konumları
                star_y = int(box_center_y - (box_hight/2))
                
                
                
                end_x = star_x + box_with #bitiş konumları
                end_y = star_y + box_hight
                            
                box_color = colors[predicted_id]
                box_color = [int(each) for each in box_color]
                            
                cv2.rectangle(frame,(star_x,star_y), (end_x,end_y), box_color,1) #1 kalınlık
                cv2.putText(frame,label,(star_x,star_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
                
                
            cv2.line(frame,(0,line),(frame.shape[1],line),(0,255,255),2)
     

 
    cv2.imshow("Detection window", frame)
            
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
   

cap.release()
cv2.destroyAllWindows()
     