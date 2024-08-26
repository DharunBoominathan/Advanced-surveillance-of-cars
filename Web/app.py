# app.py (Flask application)


from flask import Flask, render_template, request
import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import os


from dataset import torch, os, LocalDataset, transforms, np, get_class, num_classes, preprocessing, Image, m, s
from config import *

from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import resnet, vgg

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import tensorflow

from matplotlib import pyplot as plt
from numpy import unravel_index
import gc
import argparse
import pandas as pd 
import torch

import cv2
from PIL import Image 
from IPython.display import display
 


pt='G:/Final project/Car_Detection/'
img_path='G:/Final project/Web/cars/'

app = Flask(__name__, static_url_path='/cars', static_folder='G:/Final project/Web/')

# Load YOLO model
model = YOLO(pt+'yolov8n.pt')
# Load class names
with open(pt+'classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Initialize SORT tracker
tracker = Sort(max_age=20)

# Define the top margin for ROI
top_margin = 0

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")


# Function to process the video and save car images
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    unique = []
    frame_skip_interval = 5
    frame_count = 0

    directory_path = 'cars'
   # delete_files_in_directory(directory_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip detection for some frames
        if frame_count % frame_skip_interval != 0:
            continue

        roi = frame[top_margin:, :]

        detections = np.empty((0, 5))
        result = model(roi, stream=1)

        for info in result:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                classindex = box.cls[0]
                conf = math.ceil(conf * 100)
                classindex = int(classindex)
                objectdetect = classnames[classindex]

                if objectdetect == 'car' and conf > 60:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    y1 += top_margin  # Adjust y-coordinate to original frame
                    y2 += top_margin
                    new_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, new_detections))

        track_result = tracker.update(detections)

        for results in track_result:
            x1, y1, x2, y2, id = results
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            if id not in unique:
                car_image = frame[y1:y2, x1:x2]
                cv2.imwrite(f"cars/{id}.jpg", car_image)
                unique.append(id)

    cap.release()

def clean_string(s):
    s = s.lower().replace('_', ' ').replace('\\', '')
    s = s.split()
    s.pop()
    s=' '.join(s)
    print(s)
    return s

def process_image(img_path):
    mean=torch.tensor([2.2810, 2.2651, 2.2317]) 
    std_dev=torch.tensor([8.9741e+08, 8.9168e+08, 8.7895e+08])
    print(m,s)
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std_dev)])
    
    classes = {"num_classes": len(num_classes)}
    resnet152_model = resnet.resnet152(pretrained=False, **classes)
    model_name="resnet152"
    model2=resnet152_model

    class MyModel(torch.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            # Define your model architecture here

        def forward(self, x):
            # Define the forward pass of your model
            return x

    # Define the file path where the model is saved
    model_path = "results/resnet152/resnet152.pt"

    # Create an instance of your model
    model2 = MyModel()

    def test_sample(image_path, model=model2, model_name=model_name):
        im = Image.open(image_path).convert("RGB")
        im = transform(im)

        if USE_CUDA and torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        x = Variable(im.unsqueeze(0))

        if USE_CUDA and torch.cuda.is_available():
            x = x.cuda()
            pred = model(x).data.cuda().cpu().numpy().copy()
        else:
            pred = model(x).data.numpy().copy()

        print (pred)

        idx_max_pred = np.argmax(pred)
        idx_classes = idx_max_pred % classes["num_classes"]
        print(get_class(idx_classes))
        return get_class(idx_classes)
    
    df = pd.DataFrame(columns=['images','name'])
    files = os.listdir(img_path)
    files.sort()

    for i in files:
        print(test_sample(img_path+i))
        df.loc[len(df.index)]=[i,clean_string(test_sample(img_path+i))]

    return df

def search(df,inp):
    val=tx_val
    print(val)
    is_present = df['name'].str.contains(val, case=False)
    plen=is_present.sum()
    res=df[is_present]
    print(res)
    for i in range(plen):
        if not res.empty:
         result_value = res['images'].values[i]
         print(result_value)
         res=res['name'].values[i]
         return (result_value,res)
    
        # creating a object 
        im = Image.open(inp+result_value) 
        display(im)
    else:
        return ("Value not found.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video_route():
    video_file = request.files['video_file']
    global tx_val
    tx_val = request.form['text_input']
    video_path = pt + video_file.filename
    video_file.save(video_path)

    # Create directory to store car images
    if not os.path.exists('cars'):
        os.makedirs('cars')

    # Process the video and save car images
    process_video(video_path)

    # Get list of saved car images
    car_images = [f'cars/{file}' for file in os.listdir('cars')]

    return render_template('result.html', car_images=car_images)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    global df2
    df2=process_image(img_path) 
    return render_template('output.html', dataframe=df2)
    

@app.route('/search_image', methods=['POST'])
def search_image_route(): 
    try:
        ans,res=search(df2,img_path)
        ans=f'cars/{ans}'
        return render_template('output2.html', txt=ans, res=res)
    except:
        res=search(df2,img_path)
        return render_template('output2.html',res=res)


if __name__ == '__main__':
    app.run(debug=True)
