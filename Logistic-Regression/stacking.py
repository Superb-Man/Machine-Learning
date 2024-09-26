import numpy as np
import pandas as pd
from lr import LogisticRegression
from scoring import accuraci, sensitivity, specificity, precision, f1_score
from bagging import BaggingClassifer

class StackingClassifier :
    def __init__(self, bagging_models, meta_classifiier = None,alpha = 0.0001, eps = 0.00001, n_iter = 1000, l2_lambda = 1, l1_lambda = 1, regularizerType = None):
        self.bagging_models = bagging_models
        if meta_classifiier is None:
            meta_classifier = LogisticRegression(
                                    alpha , 
                                    eps, 
                                    n_iter, 
                                    l2_lambda, 
                                    l1_lambda,
                                    regularizerType
                                )
        self.meta_classifier = meta_classifier
    
    def fit(self, X, Y):
        """Fit the bagging models for validation set
         add preditions of the bagging models to the feature set as columns
         then train the meta classifier on the old + new features
        """
        # convert the X,Y to numpy array if the input is dataframe
        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X
        Y = Y.to_numpy().astype(float) if type(Y) == pd.DataFrame else Y



        # converting the predictions of the bagging models into features so transpositing
        prediction_features = np.array([model.predict(X) for model in self.bagging_models]).T
        X_new = np.hstack((X, prediction_features))  

        # train the meta classifier with validation set
        self.meta_classifier.fit(X_new, Y)

    def predict(self, X,rtype = 'binary') :
        "Predict the output of the meta classifier"
        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X

        prediction_features = np.array([model.predict(X) for model in self.bagging_models]).T
        X_new = np.hstack((X, prediction_features))
        return self.meta_classifier.predict(X_new,rtype)


        
