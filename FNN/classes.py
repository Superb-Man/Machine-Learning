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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batchsize = number of samples for now
# num_units = next layer neurons/num_of_features
# x = input data
# w = weights
# grad_output = gradient of loss with respect to the output of this layer
class Optimizer:
    def update(self,layer,weights,grad_weights):
        raise NotImplementedError
    def step(self, layers):
        raise NotImplementedError
    
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output,optimizer:Optimizer):
        raise NotImplementedError

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        raise NotImplementedError

    

class Activation(Layer):
    def forward(self, input):
        pass
    
    def backward(self,grad_output,optimizer:Optimizer):
        pass
        


########################################################## CLASS DEFINITIONS ##########################################################


class Dropout(Layer):
    def __init__(self, dropProb=0.2):
        self.dropProb = dropProb
        self.mask = None
    
    def forward(self, input):

        self.mask = np.random.rand(*input.shape) > self.dropProb
        return input * self.mask / (1 - self.dropProb)
    
    def backward(self, grad_output,optimizer:Optimizer):
        return grad_output * self.mask / (1 - self.dropProb)


    
class ReLU(Activation):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels, width, height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        
        # f(x) = max(0,x)
        self.input = input
        return np.maximum(0,input)
    
    # grad_input = dx 
    def backward(self,grad_output,optimizer:Optimizer):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels, width, height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        # f'(x) = 1 if x > 0 else 0
        grad_input = grad_output * np.where(self.input > 0, 1, 0)

        return grad_input
    
class Sigmoid(Activation):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels, width, height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input

        # f(x) = 1 / (1 + e^-x)
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self,grad_output,optimizer:Optimizer):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels, width, height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        # back-propagate through the sigmoid function
        # f'(x) = grad_output * f(x) * (1 - f(x))
        gard_input = grad_output * self.output * (1 - self.output) , None , None

        return gard_input
    


class SoftMax(Activation):
    
    EPSILON = 1e-8
    MAX = 100

    def forward(self, input):
        # Softmax formula
#       # f(x_i) = e^x_i / sum(exp(x_j))
        input = np.array(input) if not isinstance(input, np.ndarray) else input
        
        input_clipped = np.copy(input)
        input_clipped[input_clipped > SoftMax.MAX] = SoftMax.MAX
        input_clipped[input_clipped < SoftMax.EPSILON] = SoftMax.EPSILON

        exp_values = np.exp(input_clipped - np.max(input_clipped, axis=-1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        return self.output

    def backward(self, grad_output,optimizer:Optimizer):
        # if i == j
        # dL/dx_i = f(x_i) * (1 - f(x_i)) * dL/df(x_i)
        # if i != j
        # dL/dx_i = -f(x_i) * sum(f(x_j) * dL/df(x_j))

        # combining both 
        # grad_i = dL/df(x_i)
        # grad_j = dL/df(x_j)
        # dL/dx_i = f(x_i) * (grad_i - sum(grad_j * f(x_j))) 

        grad_input = self.output * (grad_output - np.sum(self.output * grad_output, axis=-1, keepdims=True))
        return grad_input 
    



# formula for dense layer
# forward propagation
# self.output = activation(self.input * self.weights)

# backpropagation
# grad_output (from next layer) ================> dL/dz = dL/da * da/dz    

# grad_weights = input.T * grad_output =========> dL/dweights = x.T * dL/dz
# grad_bias = sum(grad_output)  ================> dL/db = sum(dL/dz)
# grad_input = grad_output * weights.T =========> dL/dx = dL/dz * w.T

class Dense(Layer):
    '"Dense layer is a fully connected layer"'
    "'input_size: number of input features"
    "'output_size: number of output features"
    def __init__(self,input_size,output_size):        
        np.random.seed(0)
        self.id = id(self)
        a = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-a, a, (input_size, output_size))
        self.bias = np.ones(output_size)

    def forward(self,input):
        "'input: input data with shape (batch_size, input_size)'"
        "'output: output data with shape (batch_size, output_size)'"
        # covert input to numpy array
        input = np.array(input) if not isinstance(input,np.ndarray) else input

        self.input = input 
        self.output = np.dot(self.input,self.weights) + self.bias # z = x * weights + bias
        
        return self.output
    
    def backward(self,grad_output,optimizer:Optimizer):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, output_size)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, input_size)'"
        # grad_output = self.activation.backward(grad_output) if self.activation is not None else grad_output # dL/da * da/dz
        grad_weights = np.dot(self.input.T,grad_output) # dL/dweights = x.T * dL/dz
        grad_bias = np.sum(grad_output,axis = 0)        # dL/db = sum(dL/dz)

        # grad_clip = 1.
        # grad_weights = np.clip(grad_weights,-grad_clip,grad_clip)
        # grad_bias = np.clip(grad_bias,-grad_clip,grad_clip)

        grad_input = np.dot(grad_output,self.weights.T) # dL/dx = dL/dz * w.T
        layer_id = str(self.id)
        self.weights = optimizer.update(layer_id +'w',self.weights,grad_weights)
        self.bias = optimizer.update(layer_id + 'b',self.bias,grad_bias)

        # self.weights,self.bias = optimizer.update(self,self.weights,grad_weights,self.bias,grad_bias)

        return grad_input


class Flatten(Layer):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels*width*height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        self.input_shape = input.shape  
        return input.reshape(input.shape[0],-1)
    
    def backward(self,grad_output,optimizer:Optimizer):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels*width*height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        return grad_output.reshape(self.input_shape)
    

class CrossEntropyLoss(Loss):
    def one_hot(self,y_true,num_classes):
        "'y_true: true labels with shape (batch_size,)'"
        "'num_classes: number of classes'"
        return np.eye(num_classes)[y_true] # returns one hot encoded labels
    

    def forward(self,y_true,y_pred):
        "'y_true: true labels with shape (batch_size, num_classes)'"
        "'y_pred: predicted labels with shape (batch_size, num_classes)'"
        y_true = np.array(y_true) if not isinstance(y_true,np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred,np.ndarray) else y_pred

        epsilon = 1e-10

        if y_true.ndim == 1:
            y_true = self.one_hot(y_true,y_pred.shape[1])

        # clip values to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        m = y_pred.shape[0] # number of samples

        # L = -sum(y_true * log(y_pred)) / m
        loss = -np.mean(y_true * np.log(y_pred))
        return loss

        
    
    def backward(self,y_true,y_pred):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_classes)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_classes)'"
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y_pred.shape[0] # number of samples

        if y_true.ndim == 1:
            y_true = self.one_hot(y_true,y_pred.shape[1])

        # dL/dy_pred = -y_true / y_pred / m
        grad_output = -y_true / y_pred
        return grad_output
    
class MSE(Loss):
    def forward(self,y_true,y_pred):
        "'y_true: true labels with shape (batch_size, num_classes)'"
        "'y_pred: predicted labels with shape (batch_size, num_classes)'"
        y_true = np.array(y_true) if not isinstance(y_true,np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred,np.ndarray) else y_pred

        # L(y_pred, y_true) = sum((y_true - y_pred)^2) / m
        m = y_pred.shape[0]
        loss = np.sum((y_true - y_pred) ** 2) / m
        return loss

    def backward(self,y_true,y_pred):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_classes)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_classes)'"

        # dL/dy_pred = -2 * (y_true - y_pred) / m

        m = y_pred.shape[0]
        grad_output = -2 * (y_true - y_pred) / m
        return grad_output
    

class BCE_Loss(Loss):
    def forward(self,y_true,y_pred):
        "'y_true: true labels with shape (batch_size, num_classes)'"
        "'y_pred: predicted labels with shape (batch_size, num_classes)'"
        y_true = np.array(y_true) if not isinstance(y_true,np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred,np.ndarray) else y_pred

        epsilon = 1e-10

        # L = -sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)) / m
        m = y_pred.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    def backward(self,y_true,y_pred):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_classes)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_classes)'"
        epsilon = 1e-10

        m = y_pred.shape[0]

        # dL/dy_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / m
        grad_output = -((y_true / (y_pred + epsilon)) - ((1 - y_true) / (1 - y_pred + epsilon))) / m
        return grad_output
    
class BatchNormalization(Layer):
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        # Initialize gamma (scaling) and beta (shifting) parameters
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        
        # Moving averages for inference
        self.moving_mean = np.zeros(input_size)
        self.moving_variance = np.ones(input_size)
        
        # Hyperparameters
        self.momentum = momentum
        self.epsilon = epsilon
        self.id = id(self)
        self.training = True
    
    def forward(self, input):
        self.input = input

        if self.training:
            # Calculate batch mean and variance
            self.batch_mean = np.mean(input, axis=0)
            self.batch_variance = np.var(input, axis=0)
            
            # Normalize input
            self.x_hat = (input - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)
            
            # Scale and shift
            output = self.gamma * self.x_hat + self.beta
            
            # Update moving averages
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.batch_variance
        else:
            # Normalize input using moving averages
            x_hat = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
            
            # Scale and shift
            output = self.gamma * x_hat + self.beta
        
        return output

    def backward(self, grad_output, optimizer: Optimizer):
        # Gradients w.r.t. gamma and beta
        grad_gamma = np.sum(grad_output * self.x_hat, axis=0)
        grad_beta = np.sum(grad_output, axis=0)

        # Gradient of the normalized input (x_hat)
        grad_x_hat = grad_output * self.gamma
        
        # Gradients w.r.t. input (chain rule)
        batch_size = self.input.shape[0]
        grad_variance = np.sum(grad_x_hat * (self.input - self.batch_mean) * -0.5 * (self.batch_variance + self.epsilon) ** (-3/2), axis=0)
        grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=0) + grad_variance * np.sum(-2 * (self.input - self.batch_mean), axis=0) / batch_size
        grad_input = grad_x_hat / np.sqrt(self.batch_variance + self.epsilon) + grad_variance * 2 * (self.input - self.batch_mean) / batch_size + grad_mean / batch_size
        
        layer_id = str(self.id)
        self.gamma = optimizer.update(layer_id+'gamma', self.gamma, grad_gamma)
        self.beta = optimizer.update(layer_id+'beta', self.beta, grad_beta)
        
        return grad_input

    


        
    
# Stochastic Gradient Descent formula
# weights = weights - lr * dL/dweights
# bias = bias - lr * dL/dbias
class SGD(Optimizer):
    def __init__(self,lr = 0.001):
        self.lr = lr

    def update(self, layer, weights, grad_weights):
        weights -= self.lr * grad_weights
        return weights
    
    def step(self,layers,grad_output):
        for layer in reversed(layers):
            grad_output= layer.backward(grad_output)
        return grad_output

# Adam Optimizer formulas
# m = beta1 * m + (1 - beta1) * dL/dweights
# v = beta2 * v + (1 - beta2) * (dL/dweights) ** 2
# m_hat = m / (1 - beta1 ** t)
# v_hat = v / (1 - beta2 ** t)
# weights = weights - lr * m_hat / (np.sqrt(v_hat) + epsilon)
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def update(self,layer,weights,grad_weights):
        if layer not in self.m:
            self.m[layer] = np.zeros_like(weights)
            self.v[layer] = np.zeros_like(weights)
        
        self.m[layer] = self.beta1 * self.m[layer] + (1 - self.beta1) * grad_weights
        self.v[layer] = self.beta2 * self.v[layer] + (1 - self.beta2) * grad_weights ** 2

        m_hat = self.m[layer] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer] / (1 - self.beta2 ** self.t)

        weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return weights


        
    
    def step(self,layers,grad_output):
        self.t += 1
        for layer in reversed(layers):
            grad_output= layer.backward(grad_output,self)
        return grad_output
    
    def reset(self):
        self.m = {}
        self.v = {}
        self.t = 0



def accuracy_score(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def F1_score(y_true, y_pred):
    return f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')

def confusion_matrix(y_true, y_pred):
    return confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

def evaluate(x,y,model):
    y_pred = model.predict(x)
    loss = model.loss.forward(y,y_pred)

    acc = accuracy_score(y,y_pred)
    f1 = F1_score(y,y_pred)
    return acc,f1,loss

class FNN:
    def __init__(self,optimizer = Adam(),loss = CrossEntropyLoss()):
        self.layers = []
        self.optimizer = optimizer
        self.loss = loss
        self.bestModel = None
    
    def add(self,layer):
        self.layers.append(layer)

    
    
    def forward(self,input,training = True):
        for layer in self.layers:
            if training == False:
                if isinstance(layer,BatchNormalization):
                    layer.training = False
                if isinstance(layer,Dropout):
                    continue
            else :
                if isinstance(layer,BatchNormalization):
                    layer.training = True
                    
            input = layer.forward(input)
        return input
    
    def backward(self,grad_output):
        return self.optimizer.step(self.layers,grad_output)

    def train(self,x,y_true,epochs):
        x = x.to_numpy().astype(float) if not isinstance(x,np.ndarray) else x
        y_true = y_true.to_numpy().astype(float) if not isinstance(y_true,np.ndarray) else y_true
        lrs = [0.001]  
        for lr in lrs:
            self.optimizer.lr = lr
            for epoch in range(epochs):
                y_pred = self.forward(x)
                loss = self.loss.forward(y_true,y_pred)
                print(f"Epoch: {epoch}, Loss: {loss}")
                gradiant = self.loss.backward(y_true,y_pred)
                self.backward(gradiant)

            
    def batchTrain(self, x, y_true, epochs, batch_size=32):
        x = x.to_numpy().astype(float) if not isinstance(x, np.ndarray) else x
        y_true = y_true.to_numpy().astype(float) if not isinstance(y_true, np.ndarray) else y_true

        # lrs = [0.005,0.0001,0.00005,0.000001]
        lrs = [0.0001]
        
        for lr in lrs:
            self.optimizer.lr = lr
            self.optimizer.reset()

            best_f1 = 0.
            self.bestModel = FNN(optimizer=self.optimizer, loss=self.loss)
            
            for epoch in range(epochs):
                indices = np.random.permutation(len(x))
                x = x[indices]
                y_true = y_true[indices]
                
                total_loss = 0. 
                accuracy = 0.

                if len(x) % batch_size != 0:
                    batch_size += 1
                
                for i in range(0, len(x), batch_size):
                    x_batch = x[i:i + batch_size]
                    y_batch = y_true[i:i + batch_size]
                    
                    y_pred = self.forward(x_batch)
                    total_loss += self.loss.forward(y_batch, y_pred)

                    accuracy += accuracy_score(y_batch, y_pred)
                    
                    gradient = self.loss.backward(y_batch, y_pred)
                    self.backward(gradient)
                
                avg_loss = total_loss / (len(x) // batch_size)
                accuracy = accuracy / (len(x) // batch_size)
                print(f"Epoch: {epoch + 1}, LR: {lr}, Avg Loss: {avg_loss}")
                print(f"Accuracy: {accuracy}")

                # validation loss, accuracy, f1 score

                val_acc, f1_score, val_loss = evaluate(x, y_true, self)

                print(f"Validation Accuracy: {val_acc}, Validation F1 Score: {f1_score}, Validation Loss: {val_loss}")

                if f1_score > best_f1:
                    self.bestModel.layers = copy.deepcopy(self.layers)
                    best_f1 = f1_score

        return self.bestModel

                



    def predict(self,x):
        x = x.to_numpy().astype(float) if not isinstance(x,np.ndarray) else x
        return self.forward(x,training = False)
    



def trainModels(x_train, y_train, x_val, y_val, models, epochs=50):

    validation_size = 0.2
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=42)

    for model in models:
        model.batchTrain(x_train, y_train, epochs=epochs, batch_size=64)
        acc, f1, loss = evaluate(x_val, y_val, model)
        print(f"Accuracy: {acc}, F1 Score: {f1}, Loss: {loss}")

    return models


class KFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, X):
        X = np.array(X)
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, val_indices
            current = stop




################ MNIST DATA ################

def normalize(x):
    return x / 255.0






if __name__ == '__main__':

    transform = transforms.ToTensor()


    train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_data = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    x_train = train_data.data.numpy() 
    x_train = x_train.reshape(-1, 1, 28, 28) 
    x_train = normalize(x_train)

    x_test = test_data.data.numpy()
    x_test = x_test.reshape(-1, 1, 28, 28)
    x_test = normalize(x_test)

    y_train = np.eye(10)[train_data.targets.numpy()]  
    y_test = np.eye(10)[test_data.targets.numpy()]

    flatten = Flatten()
    dense1 = Dense(28*28, 256) 
    dense2 = Dense(256, 128)
    dense3 = Dense(128, 100)
    dense4 = Dense(100, 10)

    loss_function = CrossEntropyLoss()
    optimizer = Adam(lr=0.001)

    model = FNN(optimizer=optimizer, loss=loss_function)
    model.add(flatten)
    model.add(dense1)
    model.add(BatchNormalization(256))
    model.add(ReLU())
    model.add(Dropout())
    model.add(dense2)
    model.add(BatchNormalization(128))
    model.add(ReLU())
    model.add(Dropout())
    model.add(dense3)
    model.add(BatchNormalization(100))
    model.add(ReLU())
    model.add(Dropout())
    model.add(dense4)
    model.add(SoftMax())

    best = model.batchTrain(x_train, y_train, epochs=50, batch_size=64)

    y_pred = best.predict(x_test)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    f1_score = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")

