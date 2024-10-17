import numpy as np


# batchsize = number of samples for now
# num_units = next layer neurons/num_of_features
# x = input data
# w = weights
# grad_output = gradient of loss with respect to the output of this layer
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

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
        


########################################################## CLASS DEFINITIONS ##########################################################


class Dropout(Layer):
    def __init__(self, p=0.2):
        self.p = 1 - p
        self.mask = None
    
    def forward(self, input):
        np.random.seed(0)
        # generate a mask with the same shape as the input
        # mask = 1 with probability p
        # mask = 0 with probability 1 - p

        self.mask = np.random.binomial(1, self.p, size=input.shape) / self.p
        return input * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask , None , None

    
class ReLU(Activation):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels, width, height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input
        
        # f(x) = max(0,x)
        self.input = input
        return np.maximum(0,input)
    
    # grad_input = dx 
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels, width, height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        # f'(x) = 1 if x > 0 else 0
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
    
class Sigmoid(Activation):
    def forward(self,input):
        "'input: input data with shape (batch_size, num_channels, width, height)'"
        "'output: output data with shape (batch_size, num_channels, width, height)'"
        input = np.array(input) if not isinstance(input,np.ndarray) else input

        # f(x) = 1 / (1 + e^-x)
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, num_channels, width, height)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, num_channels, width, height)'"
        # back-propagate through the sigmoid function
        # f'(x) = grad_output * f(x) * (1 - f(x))
        return grad_output * self.output * (1 - self.output)
    
class SoftMax(Activation):
    
    def forward(self,input):
        # Softmax formula
        # f(x_i) = e^x_i / sum(exp(x_j))

        input = np.array(input) if not isinstance(input,np.ndarray) else input
        # prevent overflow
        input -= np.max(input, axis=-1, keepdims=True)
        self.output = np.exp(input) / np.sum(np.exp(input), axis=-1, keepdims=True)

        return self.output

    def backward(self,grad_output):

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
    def __init__(self,input_size,output_size,initializer = 'xaiver',activation = None):
        denominator = 1
        if initializer == 'xaiver':
            denominator = np.sqrt(input_size)
        
        np.random.seed(0)
        self.weights = np.random.randn(input_size,output_size) / denominator
        self.bias = np.ones(output_size)
        self.activation = activation

    def forward(self,input):
        "'input: input data with shape (batch_size, input_size)'"
        "'output: output data with shape (batch_size, output_size)'"
        # covert input to numpy array
        input = np.array(input) if not isinstance(input,np.ndarray) else input

        self.input = input 
        self.output = np.dot(self.input,self.weights) + self.bias # z = x * weights + bias

        if self.activation is not None:
            self.output = self.activation.forward(self.output) # a = activator(z)
        return self.output
    
    def backward(self,grad_output):
        "'grad_output: gradient of loss with respect to the output of this layer (batch_size, output_size)'"
        "'grad_input: gradient of loss with respect to the input of this layer (batch_size, input_size)'"
        grad_output = self.activation.backward(grad_output) if self.activation is not None else grad_output # dL/da * da/dz
        grad_weights = np.dot(self.input.T,grad_output) # dL/dweights = x.T * dL/dz
        grad_bias = np.sum(grad_output,axis = 0)        # dL/db = sum(dL/dz)

        grad_clip = 1.
        grad_weights = np.clip(grad_weights,-grad_clip,grad_clip)
        grad_bias = np.clip(grad_bias,-grad_clip,grad_clip)

        grad_input = np.dot(grad_output,self.weights.T) # dL/dx = dL/dz * w.T

        dL = [grad_input,grad_weights,grad_bias] 

        return dL


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
        return grad_output.reshape(self.input_shape) , None , None
    

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
        loss = -np.sum(y_true * np.log(y_pred)) / m
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
        grad_output = -y_true / y_pred / m
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
    

        
    
# Stochastic Gradient Descent formula
# weights = weights - lr * dL/dweights
# bias = bias - lr * dL/dbias
class SGD(Optimizer):
    def __init__(self,lr = 0.001):
        self.lr = lr
    
    # call the backward method of each layer to update the weights
    def step(self,layers,grad_output):
        for layer in reversed(layers):
            grad_output,grad_weights ,grad_bias = layer.backward(grad_output)
            if grad_weights is not None:
                layer.weights -= self.lr * grad_weights
                layer.bias -= self.lr * grad_bias
        
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
        
    def step(self, layers, grad_output):
        self.t += 1
        
        for idx, layer in enumerate(reversed(layers)):

            grad_output, grad_weights, grad_bias = layer.backward(grad_output) 

            if grad_weights is not None:

                if idx not in self.m:
                    self.m[idx] = {'w': np.zeros_like(grad_weights), 'b': np.zeros_like(grad_bias)}
                    self.v[idx] = {'w': np.zeros_like(grad_weights), 'b': np.zeros_like(grad_bias)}
                


                self.m[idx]['w'] = self.beta1 * self.m[idx]['w'] + (1 - self.beta1) * grad_weights
                self.v[idx]['w'] = self.beta2 * self.v[idx]['w'] + (1 - self.beta2) * grad_weights ** 2

                self.m[idx]['b'] = self.beta1 * self.m[idx]['b'] + (1 - self.beta1) * grad_bias
                self.v[idx]['b'] = self.beta2 * self.v[idx]['b'] + (1 - self.beta2) * grad_bias ** 2

                m_hat_w = self.m[idx]['w'] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v[idx]['w'] / (1 - self.beta2 ** self.t)

                m_hat_b = self.m[idx]['b'] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v[idx]['b'] / (1 - self.beta2 ** self.t)

                layer.weights = layer.weights - self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.bias = layer.bias - self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return grad_output

    

class FNN:
    def __init__(self,optimizer = Adam(),loss = CrossEntropyLoss()):
        self.layers = []
        self.optimizer = optimizer
        self.loss = loss
    
    def add(self,layer):
        self.layers.append(layer)
    
    def forward(self,input,training = True):
        for layer in self.layers:
            if training == False:
                if isinstance(layer,Dropout):
                    continue
            input = layer.forward(input)
        return input
    
    def backward(self,grad_output):
        return self.optimizer.step(self.layers,grad_output)

    def train(self,x,y_true,epochs):
        lrs = [0.001]
        for lr in lrs:
            self.optimizer.lr = lr
            for epoch in range(epochs):
                x = self.forward(x)
                loss = self.loss.forward(y_true,x)
                print(f"Epoch: {epoch}, Loss: {loss}")
                x = self.loss.backward(y_true,x)
                x = self.backward(x)


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


def train(x, y_true, epochs, n_splits=5):
    kfold = KFold(n_splits=n_splits)
    
    fold_losses = []
    bestModel = None
    prev_loss = np.inf

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(x)):
        print(f"\n--- Fold {fold+1} ---")
        model = FNN(optimizer=Adam(lr=0.001), loss=CrossEntropyLoss())
        model.add(Flatten())
        model.add(Dense(3*32*32, 100, activation=ReLU()))
        model.add(Dropout())
        model.add(Dense(100, 100, activation=ReLU()))
        model.add(Dropout())
        model.add(Dense(100, 4, activation=SoftMax()))
        
        # Split data into training and validation sets for the current fold
        x_train, y_train = x[train_idx], y_true[train_idx]
        x_val, y_val = x[val_idx], y_true[val_idx]

        # Train the model on the training set
        model.train(x_train, y_train, epochs)
        
        # Compute the loss on the validation set
        val_pred = model.forward(x_val, training=False)
        val_loss = model.loss.forward(y_val, val_pred)

        if val_loss < prev_loss:
            prev_loss = val_loss
            bestModel = model
        
        # Store the validation loss for this fold
        fold_losses.append(val_loss)
        print(f"Validation Loss for fold {fold+1}: {val_loss}")

    # Calculate average loss across all folds
    avg_loss = np.mean(fold_losses)
    print(f"\nAverage Validation Loss across {n_splits} folds: {avg_loss}")

    return bestModel








if __name__ == '__main__':
    sample_size = 1000
    num_classes = 4
    input_shape = (sample_size, 3, 32, 32)
    
    # generatae same random data each time
    np.random.seed(0)
    x = np.random.randn(*input_shape)
    # normalize the data
    x = (x - np.mean(x)) / np.std(x)

    np.random.seed(0)
    y_true = np.eye(num_classes)[np.random.choice(num_classes, sample_size)]  # random one-hot labels
    print(y_true.shape)


    # read x ttrain and y_train from npz file
    # data = np.load('data.npz')
    # x_train = data['x']
    # y_train = data['y_true']

    
    # Define the model layers
    # flatten = Flatten()
    # dense1 = Dense(3*32*32, 100, activation=ReLU())
    # dense2 = Dense(100, 100, activation=ReLU())
    # dense3 = Dense(100, num_classes, activation=SoftMax())

    # # Define loss function and optimizer
    # loss_function = CrossEntropyLoss()
    # # optimizer = SGD(lr=0.001)
    # optimizer = Adam(lr=0.001)

    # model = FNN(optimizer=optimizer, loss=loss_function)
    # model.add(flatten)
    # model.add(dense1)
    # model.add(Dropout())
    # model.add(dense2)
    # model.add(Dropout())
    # model.add(dense3)

    # model.train(x, y_true, epochs=100)
    # y_pred = model.predict(x)
    # print(y_pred)

    # accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    # print(f"Accuracy: {accuracy}")


    model = train(x, y_true, epochs=100, n_splits=5)

    y_pred = model.forward(x, training=False)

    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    print(f"Accuracy: {accuracy}")
