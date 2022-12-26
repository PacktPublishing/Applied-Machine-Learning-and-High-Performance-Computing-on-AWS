import json
import os
import numpy as np
import pandas as pd
import time
from PIL import Image
import requests

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import io
import json
import logging
import pickle

import neopytorch
import torchvision.transforms as T

def input_fn(request_body, content_type):
    print('--------- Deserializing the input data. ---------')
    if content_type == 'application/octet-stream':
        image_data = Image.open(io.BytesIO(request_body))
        image_transform = T.Compose([
            T.Resize(size=256),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        normalized = image_transform(image_data)
        input_data = normalized.unsqueeze(0)
        return input_data
    else:
        print('raising expception')
        raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_data, model):
    print('Generating prediction based on input parameters.')
     # predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    result = model.forward(input_data)
    print('length of result tensor: ', len(result[0]))
    return result

def output_fn(prediction_output, accept='application/json'):
    print('Serializing the generated output.')
    object_categories = {}
    result = np.squeeze(prediction_output[0].cpu().detach().numpy())
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp) 
    class_id = np.argmax(result)
    print('-------- class id -----------', class_id)
    print('-------- list directory --------', os.listdir(os.getcwd()+'/code'))
    with open("code/labels.txt", "r") as f:
        for line in f:
            key, val = line.strip().split(":")
            object_categories[key] = val.strip(" ").strip(",")
    label = object_categories[str(class_id)]
    pred = {'label': str(label), 'probability': str("{:.2f}%".format(np.max(result)*100))}
    if accept == 'application/json':
        return json.dumps(pred), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')