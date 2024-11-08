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

def accuracy_score(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
def F1_score(y_true, y_pred):
    return f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')


def plot_confusion_matrix(y_true, y_pred,lr):
    # if not os.path.exists(str(lr)):
    #     os.makedirs(str(lr))
    cm = confusion_matrix(np.argmax(y_true,axis=1), np.argmax(y_pred,axis=1))
    sns.heatmap(cm, annot=True, fmt='d')
    # plt.savefig(f'./str(lr)/confusion_matrix.png')
    plt.show()
    

def plot_train_val(train , val,name,lr):

    plt.plot(train, label='Training')
    plt.plot(val, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    # plt.savefig(f'./{lr}/{name}.png')
    plt.show()

def plot_val(val,name,lr):
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    # plt.savefig(f'./{lr}/{name}.png')
    plt.show()


def scoring(y_true, y_pred, model):
    loss = model.loss.forward(y_true,y_pred)

    acc = accuracy_score(y_true,y_pred)
    f1 = F1_score(y_true,y_pred)
    return acc,f1,loss
