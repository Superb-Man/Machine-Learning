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



def batchTrain(model,x_train,y_train,x_val,y_val,epochs = 25, batch_size=32,lr = 0.0001):
    # print(x)
    
    model.optimizer.lr = lr
    model.optimizer.reset()

    train_loss = []
    train_accuracy = []

    val_los = []
    val_accuracy = []
    val_f1 = []

    best_f1 = 0.

    model.bestModel = nn.FNN(optimizer=model.optimizer, loss=model.loss)
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        # print(x_train.shape, y_train.shape)
        
        total_loss = 0. 
        accuracy = 0.

        if len(x_train) % batch_size != 0:
            batch_size += 1
        
        for i in range(0, len(x_train), batch_size):
            x_batch = np.array(x_train[i:i + batch_size])
            y_batch = np.array(y_train[i:i + batch_size])
            
            y_pred = model.forward(x_batch)
            total_loss += model.loss.forward(y_batch, y_pred)

            accuracy += ut.accuracy_score(y_batch, y_pred)
            
            gradient = model.loss.backward(y_batch, y_pred)
            model.backward(gradient)
        
        avg_loss = total_loss / (len(x_train) // batch_size)
        accuracy = accuracy / (len(x_train) // batch_size)
        print(f"Epoch: {epoch + 1}, LR: {lr}, Avg Loss: {avg_loss}")
        print(f"Accuracy: {accuracy}")

        train_loss.append(avg_loss)
        train_accuracy.append(accuracy)

        # validation loss, accuracy, f1 score
        y_pred = nn.predict(model, x_val)

        val_acc, f1_score, val_loss = ut.scoring(y_val, y_pred, model)

        val_accuracy.append(val_acc)
        val_los.append(val_loss)
        val_f1.append(f1_score)

        print(f"Validation Accuracy: {val_acc}, Validation F1 Score: {f1_score}, Validation Loss: {val_loss}")

        if f1_score > best_f1:
            model.bestModel.layers = copy.deepcopy(model.layers)
            best_f1 = f1_score
        
        model.optimizer.lr = nn.LR_Scheduler().exSchedule(lr, epoch)

    y_pred = nn.predict(model.bestModel, x_val)
    ut.plot_confusion_matrix(y_val, y_pred,lr)
    ut.plot_train_val(train_loss, val_los, 'Loss',lr)
    ut.plot_train_val(train_accuracy, val_accuracy, 'Accuracy',lr)
    ut.plot_val(val_f1, 'F1 Score',lr)


    return model.bestModel



def LightModel():
    model = nn.FNN(optimizer=nn.Adam(), loss=nn.CrossEntropyLoss())
    model.add(nn.Flatten())
    model.add(nn.Dense(28*28,128))
    model.add(nn.BatchNormalization(128))
    model.add(nn.ReLU())
    model.add(nn.Dense(128,10))
    model.add(nn.SoftMax())

    return model

def MediumModel():
    model = nn.FNN(optimizer=nn.Adam(), loss=nn.CrossEntropyLoss())
    model.add(nn.Flatten())
    model.add(nn.Dense(28*28,256))
    model.add(nn.BatchNormalization(256))
    model.add(nn.ReLU())
    model.add(nn.Dropout(0.3))
    model.add(nn.Dense(256,64))
    model.add(nn.BatchNormalization(64))
    model.add(nn.ReLU())
    model.add(nn.Dropout(0.3))
    model.add(nn.Dense(64,10))
    model.add(nn.SoftMax())
    
    return model

def HeavyModel():
    model = nn.FNN(optimizer=nn.Adam(), loss=nn.CrossEntropyLoss())
    model.add(nn.Flatten())
    model.add(nn.Dense(28*28,512))
    model.add(nn.BatchNormalization(512))
    model.add(nn.ReLU())
    model.add(nn.Dropout(0.3))
    model.add(nn.Dense(512,256))
    model.add(nn.BatchNormalization(256))
    model.add(nn.ReLU())
    model.add(nn.Dropout(0.3))
    model.add(nn.Dense(256,128))
    model.add(nn.BatchNormalization(128))
    model.add(nn.ReLU())
    model.add(nn.Dropout(0.3))
    model.add(nn.Dense(128,64))
    model.add(nn.BatchNormalization(64))
    model.add(nn.ReLU())
    model.add(nn.Dropout(0.3))
    model.add(nn.Dense(64,10))
    model.add(nn.SoftMax())
    
    return model

def trainModels(x_train, y_train):
    x_train = x_train.to_numpy().astype(float) if not isinstance(x_train, np.ndarray) else x_train
    y_train = y_train.to_numpy().astype(float) if not isinstance(y_train, np.ndarray) else y_train

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)


    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)

    # train light model
    lrs = [0.005,0.0005,0.0003,0.0001]

    # for lr in lrs:
    # #    if not os.path.exists(str(lr)):
    # #       os.mkdir(str(lr))
    #     arch1 = LightModel()

    #     best = batchTrain(arch1, x_train,y_train,x_val,y_val, epochs=50, batch_size=75, lr=lr)
    #     best.clear()
    #     nn.saveModel(best, f"light_model_{lr}.pkl")
    for lr in lrs:
        # if not os.path.exists(str(lr)):
        #     os.mkdir(str(lr))
        
        arch2 = MediumModel()

        best = batchTrain(arch2,x_train, y_train,x_val,y_val, epochs=50, batch_size=64, lr=lr)
        best.clear()
        nn.saveModel(best, f"med_model_{lr}.pkl")
    # for lr in lrs:
    # #     st = 'heavy'+str(lr)
    # #     if not os.path.exists(st):
    #         os.mkdir(st)

    #     arch3 = HeavyModel()

    #     best = batchTrain(arch3,x_train,y_train,x_val, y_val, epochs=50, batch_size=64, lr=lr)
    #     best.clear()
    #     nn.saveModel(best, f"heavy_model_{lr}.pkl")



if __name__ == '__main__':

    transform = transforms.ToTensor()


    train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

    x_train = train_data.data.numpy() 
    x_train = x_train.reshape(-1, 1, 28, 28) 
    x_train = nn.normalize(x_train)

    y_train = np.eye(10)[train_data.targets.numpy()]  

    # print(y_train)


    trainModels(x_train, y_train)