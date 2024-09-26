import numpy as np
import pandas as pd
from lr import LogisticRegression
from scoring import accuraci, sensitivity, specificity, precision, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt

class BaggingClassifer :
    def __init__(self, n_estimators = 10, max_samples = 1.0, n_iter = 1000, alpha = 0.0001, eps = 0.00001, l2_lambda = 1, l1_lambda = 1, regularizerType = None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []
        self.params = {
            'n_iter': n_iter,
            'alpha': alpha,
            'eps': eps,
            'l2_lambda': l2_lambda,
            'l1_lambda': l1_lambda,
            'regularizerType': regularizerType
        }
    
    def fit(self, X, Y):
        # if dataframe
        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X 
        Y = Y.to_numpy().astype(float) if type(Y) == pd.DataFrame else Y

        # print('Bagging fit called')

        np.random.seed(42) # for reproducibility
        for i in range(self.n_estimators):
            # sample with replacement
            idx = np.random.choice(X.shape[0], int(self.max_samples * X.shape[0]), replace=True)
            model = LogisticRegression(**self.params)
            # print(np.unique(X[idx], axis=0).shape)
            model.fit(X[idx], Y[idx])
            self.models.append(model)

    def draw_violin_plot(self, X, Y):
        # Draw violin plots for each performance metric for the 9 bagging LR learners
        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X 
        Y = Y.to_numpy().astype(float) if type(Y) == pd.DataFrame else Y

        metrics_data = {
            'Learner': [],
            'Metric': [],
            'Value': []
        }

        # Calculate metrics for each model
        for i, model in enumerate(self.models):
            y_pred = model.predict(X)
            Accuracy = accuraci(y_pred, Y)
            Sensitivity = sensitivity(y_pred, Y)
            Specificity = specificity(y_pred, Y)
            Precision = precision(y_pred, Y)
            F1 = f1_score(y_pred, Y)
            y_pred = model.predict(X, rtype='sigmoid')
            Auroc = roc_auc_score(Y, y_pred)
            Aupr = average_precision_score(Y, y_pred)

            # Store metrics for plotting
            metrics_data['Learner'].extend([i+1] * 7)
            metrics_data['Metric'].extend(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'AUROC', 'AUPR'])
            metrics_data['Value'].extend([Accuracy, Sensitivity, Specificity, Precision, F1, Auroc, Aupr])

        metrics_df = pd.DataFrame(metrics_data)
        # mean and std_deviation for bagging LR learners

        mean_std = metrics_df.groupby('Metric').agg({'Value': ['mean', 'std']})

        print("Averege and standard deviation for bagging LR learners")
        print(mean_std)
        print()

        # plt.figure(figsize=(14,10))
        # sns.violinplot(x='Metric', y='Value', data=metrics_df, inner='box')
        # plt.title('Performance Metrics for Bagging Logistic Regression Learners')
        # plt.show()

        # Plot violin plot for each metric
        for metric in ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'AUROC', 'AUPR']:
            plt.figure(figsize=(3, 3))
            sns.violinplot(data=metrics_df[metrics_df['Metric'] == metric], x='Metric', y='Value',inner='box')
            plt.title(f'{metric} for 9 Bagging LR learners')
            plt.show()

    
    def predict(self, X,rtype='binary') :

        X = X.to_numpy().astype(float) if type(X) == pd.DataFrame else X
        y_pred = np.zeros(X.shape[0])
        for model in self.models:
            y_pred += model.predict(X,rtype)
        y_pred /= self.n_estimators
        
        return np.array([1 if i >= 0.5 else 0 for i in y_pred]) if rtype == 'binary' else y_pred