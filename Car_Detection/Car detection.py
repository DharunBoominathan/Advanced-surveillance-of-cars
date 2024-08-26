import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone


cap = cv2.VideoCapture('new.mp4')
model = YOLO('yolov8n.pt')


car_images = {}

top_margin=0

classnames  = []
with open('classes.txt','r') as f:
    classnames = f.read().splitlines()

tracker = Sort(max_age=20)
counter = []
unique=[]

frame_skip_interval = 2
frame_count = 0

while 1:
    ret,frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip detection for some frames
    if frame_count % frame_skip_interval != 0:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        continue

    roi = frame[top_margin:,:]


    detections = np.empty((0,5))
    result = model(roi, stream=1)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]

            if objectdetect == 'car' and conf >60:
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                y1 += top_margin  # Adjust y-coordinate to original frame
                y2 += top_margin
                new_detections = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,new_detections))

                # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                # cvzone.putTextRect(frame,f'{objectdetect} {conf}%',
                #                    [x1+8,y1-12],thickness=2,scale=1.5)
                # print(classindex)

    track_result = tracker.update(detections)

    

    for results in track_result:
        x1,y1,x2,y2,id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2),int(id)

        w,h = x2-x1,y2-y1
        cx,cy = x1+w//2 , y1+h//2
        
        if(id not in unique):
            car_image=frame[y1:y2,x1:x2]
            cv2.imwrite(f"{id}.jpg",car_image)
            unique.append(id)

        cv2.circle(frame,(cx,cy),6,(0,0,255),-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cvzone.putTextRect(frame,f'{id}',
                           [x1+8,y1-12],thickness=2,scale=1.5)
    

        
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

