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


if __name__ == '__main__':
    transform = transforms.ToTensor()

    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    x_test = test_data.data.numpy()
    y_test = test_data.targets.numpy()

    x_test = nn.normalize(x_test)
    y_test = np.eye(10)[test_data.targets.numpy()]

    model = nn.loadModel('best_model.pkl')

    y_pred = nn.predict(model, x_test)
    print(y_pred.shape)
    print(y_test.shape)

    # nn. accuracy
    print('Accuracy: ', ut.accuracy_score(y_test, y_pred))
    print('F1 Score: ', ut.F1_score(y_test, y_pred))

    ut.plot_confusion_matrix(y_test, y_pred, lr = 0)