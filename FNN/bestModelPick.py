import utils as ut
import classes as nn

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from typing import List
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import torch
from torchvision import datasets, transforms
import copy
import os


if __name__ == '__main__':

    bestModel = None
    best_f1 = 0.
    # 0.0001,0.0003,0.0005,005
    light_models = ['light_model_0.0001.pkl', 'light_model_0.0003.pkl', 'light_model_0.0005.pkl', 'light_model_0.005.pkl']
    med_models = ['med_model_0.0001.pkl', 'med_model_0.0003.pkl', 'med_model_0.0005.pkl', 'med_model_0.005.pkl']
    heavy_models = ['heavy_model_0.0003.pkl', 'heavy_model_0.0005.pkl', 'heavy_model_0.005.pkl', 'heavy_model_0.0001.pkl']

    # load validation data from npy file
    x_val = np.load('x_val.npy')
    y_val = np.load('y_val.npy')

    for model in light_models:
        model = nn.loadModel(model)
        y_pred = nn.predict(model,x_val)
        f1 = ut.F1_score(y_val, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            bestModel = copy.deepcopy(model)

    for model in med_models:
        model = nn.loadModel(model)
        y_pred = nn.predict(model,x_val)
        f1 = ut.F1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            bestModel = copy.deepcopy(model)

    for model in heavy_models:
        model = nn.loadModel(model)
        y_pred = nn.predict(model,x_val)
        f1 = ut.F1_score(y_val, y_pred)
        # print(f1)
        if f1 > best_f1:
            best_f1 = f1
            bestModel = copy.deepcopy(model)

    nn.saveModel(bestModel, 'best_model.pkl')