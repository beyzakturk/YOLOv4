# YOLOv4
![Image](https://brunocapuano.files.wordpress.com/2020/06/yolov4-realtime-demo.gif)

## What is YOLO?
YOLO stands for You Only Look Once. YOLO is a state-of-the-art, real-time object detection system. It was developed by Joseph Redmon. It is a real-time object recognition system that can recognize multiple objects in a single frame. YOLO has evolved into newer versions over time, viz., YOLOv2, YOLOv3, and YOLOv4.

YOLO uses a totally different approach than other previous detection systems. It applies a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

The basic idea of YOLO is exhibited in the figure below. YOLO divides the input image into an S × S grid and each grid cell is responsible for predicting the object centered in that grid cell.

![Image](https://miro.medium.com/max/1200/0*Okuwq93g3v13CShN)

Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.

![Image](https://miro.medium.com/max/1400/0*IhbtJNWpPG1PgTRk)

YOLO model has several advantages over classifier-based systems. It can recognize multiple objects in a single frame. It looks at the whole image at test time so its predictions are informed by the global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast, more than 1000x faster than R-CNN and 100x faster than Fast R-CNN. The YOLO design enables end-to-end training and real-time speeds while maintaining high average precision.

## What is YOLOv4?
YOLOv4 is an object detection algorithm that is an evolution of the YOLOv3 model. The YOLOv4 method was created by Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. It is twice as fast as EfficientDet with comparable performance. In addition, AP (Average Precision) and FPS (Frames Per Second) in YOLOv4 have increased by 10% and 12% respectively compared to YOLOv3. YOLOv4’s architecture is composed of CSPDarknet53 as a backbone, spatial pyramid pooling additional module, PANet path-aggregation neck, and YOLOv3 head.

YOLOv4 uses many new features and combines some of them to achieve state-of-the-art results: 43.5% AP (65.7% AP50) for the MS COCO dataset at a real-time speed of ~65 FPS on Tesla V100. Following are the new features used by YOLOv4:
- Weighted-Residual-Connections (WRC)
- Cross-Stage-Partial-connections (CSP)
- Cross mini-Batch Normalization (CmBN)
- Self-adversarial-training (SAT)
- Mish activation
- Mosaic data augmentation
- DropBlock regularization
- Complete Intersection over Union loss (CIoU loss)

![Image](https://production-media.paperswithcode.com/methods/new_ap.jpg)


