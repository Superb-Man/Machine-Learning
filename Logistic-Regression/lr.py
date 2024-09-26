import numpy as np
from scoring import accuraci, sensitivity, specificity, precision, f1_score
from diff import sigmoid, loss, gradient, ridgeRegularization, lassoRegularization, ridgeGrad, lassoGrad
import pandas as pd

class LogisticRegression:
    def __init__(self, alpha = 0.0001, eps = 0.00001, n_iter = 1000, l2_lambda = 1, l1_lambda = 1 , regularizerType = None, theta = None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.eps = eps
        self.theta = theta
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.regularizerType = regularizerType
    
    def fit(self, X, Y) :

        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X
        Y = Y.to_numpy().astype(float) if type(Y) == pd.DataFrame else Y

        X = np.insert(X, 0, 1, axis=1) # add bias term to the first column
        Y = Y.reshape(X.shape[0], 1) # reshape Y to be a column vector


        # initialize theta , X and W needs to be multiplied
        # So theta is a column vector, row_X = column_theta
        if self.theta is None:
            self.theta = np.zeros((X.shape[1], 1))

        # print(X.shape, Y.shape, self.theta.shape)
    
        iteration = 0
        prev_cost = -np.inf

        regularizer = {
            'ridge': ridgeRegularization,
            'lasso': lassoRegularization
        }

        gradients = {
            'ridge': ridgeGrad,
            'lasso': lassoGrad
        }

        while iteration < self.n_iter:
            # calculate h(x) = sigmoid(theta^T * X + b)
            h = sigmoid(np.dot(X, self.theta))

            regularizerCost = 0

            if self.regularizerType is not None:
                regularizerCost = regularizer[self.regularizerType](X.shape[0], self.theta, self.l2_lambda)
            cost = loss(Y,h) + regularizerCost

            if cost - prev_cost < self.eps:
                print(cost - prev_cost)
                print(f"Stopping early at iteration {iteration} due to minimal change in cost.")
                break

            prev_cost = cost

            # print("cost at iteration ", iteration, " : ", cost)

            grad = gradient(X, Y, h)
            if self.regularizerType is not None:
                grad = gradients[self.regularizerType](X.shape[0], self.l2_lambda, self.theta, grad)
            

            self.theta += self.alpha * grad
            iteration += 1
    
    def predict(self,X,rtype='binary') :
        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X
        X = np.insert(X, 0, 1, axis=1)   
        # print(X.shape, self.theta.shape)
        h = sigmoid(np.dot(X, self.theta))
        if rtype == 'sigmoid':
            return np.array(h.flatten())
        else:
            return np.array([1 if i >= 0.5 else 0 for i in h])
