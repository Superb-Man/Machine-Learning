# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# # !kaggle competitions download -c dogs-vs-cats
# !kaggle datasets download -d tongpython/cat-and-dog


import zipfile
ref = zipfile.ZipFile('/content/cat-and-dog.zip')
ref.extractall('/content')
ref.close()

# ref = zipfile.ZipFile('/content/test1.zip')
# ref.extractall('/content/')


# ref = zipfile.ZipFile('/content/train.zip')
# ref.extractall('/content/')

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import glob
from PIL import Image

import copy
from tqdm import tqdm





class datasetloader(Dataset):
    def __init__(self,dl,cl,path, transform=None):
        self.classes = ['dogs','cats']
        # print(self.classes)
        self.file_list = []
        for i in range(len(dl)) :
            if not dl[i].startswith('.') and not dl[i].startswith('_'):
                self.file_list.append([0,self.classes[0],path+'/dogs/'+dl[i]])
        for i in range(len(cl)) :
            if not cl[i].startswith('.') and not cl[i].startswith('_'):
                self.file_list.append([1,self.classes[1],path+'/cats/'+cl[i]])
        print("File list  ", self.file_list)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im, classCategory



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        # (w-f+2p)/s + 1
        self.sigmoid = nn.Sigmoid()
        #(256,3,100,100)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding= 1,bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )
        ##(256,16,50,50)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        ##(256,32,25,25)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1,bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        #(256,64,12,12)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1,bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        #(256,128,6,6)
        self.fc1 = nn.Linear(128*6*6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x



#function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32


# define training and test data directories
data_dir = '/content/'
os.chdir(data_dir)
listd = os.listdir(data_dir)
print(listd)
train_dir = 'training_set/training_set'
test_dir = 'test_set/test_set'
Val_dir = 'validation'

try:
    os.mkdir(Val_dir)
except OSError as error:
    print(error)


os.path.join(data_dir,Val_dir)
print(os.listdir(data_dir))
print("Lara Loves Linux")

# print the current directory
print("Current working directory is:", os.getcwd())


train_dog_List = os.listdir(data_dir+train_dir+'/dogs')
train_cat_list = os.listdir(data_dir+train_dir+'/cats')


##Validationset
d = int(len(train_dog_List)*0.75)
c = int(len(train_cat_list)*0.75)

val_doglist = train_dog_List[d:len(train_dog_List)]
val_catlist = train_cat_list[c:len(train_cat_list)]
train_dog_List = train_dog_List[0:d]
train_cat_list = train_cat_list[0:c]
print(len(train_dog_List),train_cat_list[0])
test_dog_list = os.listdir(data_dir+test_dir+'/dogs')
test_cat_list = os.listdir(data_dir+test_dir+'/cats')
print(len(val_doglist),"here i am status : 200")





#create transformers and Data Augementation
image_size = (100, 100)
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
train_transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.RandomRotation(degrees=20),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.005),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.Normalize(mean, std)])
test_transforms = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean,std)])


 ## read data set using the custom class
train_dataset = datasetloader(train_dog_List,train_cat_list, train_dir,transform=train_transform)
print(len(train_dataset))
test_dataset = datasetloader(test_dog_list,test_cat_list,test_dir, transform=test_transforms)
print(len(test_dataset))
Val_dataset = datasetloader(val_doglist,val_catlist,train_dir, transform=test_transforms)

print("All sizes : ", len(train_dataset),len(Val_dataset))

## load data using utils
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
     num_workers=num_workers, shuffle=True, drop_last = True)
Val = torch.utils.data.DataLoader(Val_dataset, batch_size=batch_size,
     num_workers=num_workers,shuffle = True,drop_last = True)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
     num_workers=num_workers,shuffle = True)


##################################   TRAINING   ########################################

accuracy_list = []


def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = []
    for (data, target) in Val:
        data, target = Variable(data), Variable(target.long())
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train(model):

    epoch_num = 50
    min_val_loss = 5
    global best_model
    best_model = None
    train_loss = []
    for epoch in range(epoch_num) :
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
            # if batch_idx % 10 and  epoch >5:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
        model.train()
        val_loss = get_val_loss(model, Val)
        # print("Val loss : ", val_loss)
        if epoch + 1 > 4 and val_loss < min_val_loss:
            # print("here")
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            # print(best_model)

        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))

model_cnn2 = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_cnn2.parameters(), lr=0.001)
print('Number of parameters: {}'.format(get_n_params(model_cnn2)))
train(model_cnn2)



#################################  TESTING   ########################################   
def test(model):
    for i in range(10) :
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in Val:

            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(Val.dataset)
        accuracy = 100. * correct / len(Val.dataset)
        accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))

test(best_model)