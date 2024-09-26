import numpy as np
from sklearn.preprocessing import StandardScaler


# sigmoid function
def sigmoid(z) :
    return 1.0 / (1.0 + np.exp(-z))

# loss function for all data points
def loss(y, y_pred) :
    # loss = -1/n * sum(y*log(h(x)) + (1-y)*log(1-h(x)))
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
# gradient of loss function for all data points
def gradient(X, y, y_pred) :
    # gradient = (y-h(x))x
    return np.dot(X.T,y - y_pred)

def ridgeRegularization(m,theta,l2_lambda) :
    # regularization = (1/2m) * lambda * sum(|theta|^2)
    return (l2_lambda / (2 * m)) * np.sum(np.square(theta[1:])) 

def lassoRegularization(m,theta,l1_lambda) :
    # regularization = (1/2m) * lambda * sum(|theta|)
    return (l1_lambda / (2 * m)) * np.sum(np.abs(theta[1:]))

def ridgeGrad(m, l2_lambda, theta, grad) :
    grad[1:] += (l2_lambda / m) * theta[1:]
    return grad

def lassoGrad(m, l1_lambda, theta, grad) :
    grad[1:] += (l1_lambda / m) * np.sign(theta[1:])
    return grad