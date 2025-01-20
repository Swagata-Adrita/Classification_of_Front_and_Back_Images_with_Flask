#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:28:49 2025

@author: swagata
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = models.resnet18(pretrained=True)
# Load your trained model
model_path = "resnet18_model_2.pth"
model.load_state_dict(torch.load(model_path))

#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(image_path, model):
    model.eval()
    image = Image.open(image_path)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(input_batch)
        _, predicted_class = output.max(1)


#predicted_class = output.max(1)
        predicted_class.cpu().numpy()

# Map the predicted class to the class name
        class_names = ['pm-back', 'pm-full']  # Make sure these class names match your training data
        predicted_class_name = class_names[predicted_class.item()]
    
        return predicted_class_name


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)


