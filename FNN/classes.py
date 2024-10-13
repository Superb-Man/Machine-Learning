import numpy as np
# import logistic_regression from sklearn
from sklearn.linear_model import LogisticRegression

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        raise NotImplementedError
    
class Optimizer:
    def step(self, layers):
        raise NotImplementedError
    

class Activation:
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, input, grad_output):
        raise NotImplementedError
    
class ReLU(Activation):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels, width, height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        self.input = input
        return np.maximum(0,input)
    
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels, width, height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
    
class Sigmoid(Activation):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels, width, height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels, width, height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        # back-propagate through the sigmoid function
        # derivative of sigmoid function is sigmoid(x) * (1 - sigmoid(x))
        return grad_output * self.output * (1 - self.output)



class Layer:
    # batchsize = number of samples for now
    # num_units = next layer neurons/num_of_features
    # x = input data
    # w = weights
    # grad_output = gradient of loss with respect to the output of this layer
    def forward(self, input):
        # z = x * w.T => (batch_size, num_features) * (num_features, num_units) = (batch_size, num_units)
        raise NotImplementedError
    
    def backward(self, grad_output):
        # del = dL/da * da/dz * dz/dw  
        # dw = x.T * grad_output => (num_features, batch_size) * (batch_size, num_units) = (num_features, num_units)
        # dx = grad_output * w => (batch_size, num_units) * (num_units, num_features) = (batch_size, num_features)
        raise NotImplementedError
    
class Dense(Layer):
    '"Dense layer is a fully connected layer"'
    "'input_size: number of input features"
    "'output_size: number of output features"
    def __init__(self,input_size,output_size,initializer = 'xaiver',activation = None):
        denominator = 1
        if initializer == 'xaiver':
            denominator = np.sqrt(input_size)
        
        # add 1 for bias term
        self.weights = np.random.randn(input_size+1,output_size) / denominator
        self.activation = activation

    def forward(self,input):
        "'input: input data with shape (batch_size, input_size)'"
        "'output: output data with shape (batch_size, output_size)'"
        # covert input to numpy array
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        # check if input has bias term
        if input.shape[1] == self.weights.shape[0] -1:
            # add an extra column of ones for bias term
            input = np.insert(input,0,1,axis = 1)

        self.input = input
        # print(self.weights.shape,self.input.shape)

        self.output = np.dot(self.input,self.weights)

        if self.activation is not None:
            self.output = self.activation.forward(self.output)
        return self.output
    
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, output_size)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, input_size)'"
        grad_output = self.activation.backward(grad_output) if self.activation is not None else grad_output
        grad_weights = np.dot(self.input.T,grad_output)

        # donot backpropagate through bias term
        grad_input = np.dot(grad_output,self.weights[:-1].T)

        return grad_input , grad_weights


class Flatten(Layer):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels*width*height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        self.input_shape = input.shape  
        return input.reshape(input.shape[0],-1)
    
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels*width*height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        return grad_output.reshape(self.input_shape) , None
    

class CrossEntropyLoss(Loss):
    def forward(self,y_true,y_pred):
        "'y_true: true labels with shape (batch_size, num_classes)'"
        "'y_pred: predicted labels with shape (batch_size, num_classes)'"
        y_true = np.array(y_true) if not isinstance(y_true,np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred,np.ndarray) else y_pred

        epsilon = 1e-10

        # clip values to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        m = y_pred.shape[0] # number of samples

        # calculate loss loglikelihood
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

        
    
    def backward(self,y_true,y_pred):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_classes)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_classes)'"
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y_pred.shape[0] # number of samples

        # calculate gradient of loss with respect to the output of this layer
        grad_output = (-y_true / y_pred) / m
        return grad_output
        
    
class SGD(Optimizer):
    def __init__(self,lr = 0.001):
        self.lr = lr
    
    # call the backward method of each layer to update the weights
    def step(self,layers,grad_output):
        for layer in reversed(layers):
            grad_output,grad_weights = layer.backward(grad_output)
            if grad_weights is not None:
                layer.weights -= self.lr * grad_weights
        
        return grad_output


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # momentum for each layer
        self.v = {}  # variance for each layer
        self.t = 0
        
    def step(self, layers, grad_output):
        self.t += 1
        
        for idx, layer in enumerate(reversed(layers)):     
            grad_output, grad_weights = layer.backward(grad_output)       
            if grad_weights is not None:
                # print(idx)
                if idx not in self.m:
                    self.m[idx] = np.zeros_like(grad_weights)
                    self.v[idx] = np.zeros_like(grad_weights)
                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad_weights
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad_weights ** 2)
                

                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                layer.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return grad_output

    

class FNN:
    def __init__(self,optimizer = Adam(),loss = CrossEntropyLoss()):
        self.layers = []
        self.optimizer = optimizer
        self.loss = loss
    
    def add(self,layer):
        self.layers.append(layer)
    
    def forward(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self,grad_output):
        return optimizer.step(self.layers,grad_output)

    def train(self,x,y_true,epochs):
        lrs = [0.005,0.0025,0.001,0.0005,0.0001]
        for lr in lrs:
            self.optimizer.lr = lr
            for epoch in range(epochs):
                x = self.forward(x)
                loss = self.loss.forward(y_true,x)
                print(f"Epoch: {epoch}, Loss: {loss}")
                x = self.loss.backward(y_true,x)
                x = self.backward(x)

    def predict(self,x):
        return self.forward(x)



if __name__ == '__main__':
    sample_size = 1000
    num_classes = 4
    input_shape = (sample_size, 3, 32, 32)
    
    # generatae same random data each time
    np.random.seed(0)
    x = np.random.randn(*input_shape)
    # normalize the data
    x = (x - np.mean(x)) / np.std(x)

    y_true = np.eye(num_classes)[np.random.choice(num_classes, sample_size)]  # random one-hot labels
    print(y_true.shape)

    x_train = x[:800]
    y_train = y_true[:800]

    x_test = x[800:]
    y_test = y_true[800:]

    
    # Define the model layers
    flatten = Flatten()
    dense1 = Dense(3*32*32, 100, activation=ReLU())
    dense2 = Dense(100, 50, activation=ReLU())
    dense3 = Dense(50, 4, activation=Sigmoid())

    # Define loss function and optimizer
    loss_function = CrossEntropyLoss()
    optimizer = SGD(lr=0.001)
    optimizer = Adam(lr=0.001)

    model = FNN(optimizer=optimizer, loss=loss_function)
    model.add(flatten)
    model.add(dense1)
    model.add(dense2)
    model.add(dense3)

    model.train(x_train, y_train, epochs=10)
    y_pred = model.predict(x_test)
    print(y_pred)

    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Accuracy: {accuracy}")
