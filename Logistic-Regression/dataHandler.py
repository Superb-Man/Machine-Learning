import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def scale_data(features,cols, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('Invalid scaling method')
    features[cols] = scaler.fit_transform(features[cols])





def load_data(Features,Labels,target_column,scaling_method='standard', nofFeatures=20) :
    # duplicated row indexes
    duplicates = Features[Features.duplicated()].index
    Features.drop(duplicates, inplace=True)
    Labels.drop(duplicates, inplace=True)

    # drop the row if target column is missing
    missing_targets = Labels[Labels.isnull()].index
    Features.drop(missing_targets, inplace=True)
    Labels.drop(missing_targets, inplace=True)


    Features.fillna(Features.mean(numeric_only=True), inplace=True)

    # fill missing values with mode for object columns
    for column in Features.columns:
        if Features[column].dtype == 'object':
            Features[column].fillna(Features[column].mode()[0], inplace=True)
    # Label Encoding the target column
    Labels = pd.DataFrame(LabelEncoder().fit_transform(Labels),columns=[target_column])

    # print(Features.isnull().sum())

    # One hot encoding the categorical columns
    for column in Features.columns:
        if Features[column].dtype == 'object':
            Features[column] = Features[column].astype('category')
    Features = pd.get_dummies(Features, drop_first=True)

    # scale the data
    scale_data(
        Features,
        Features.columns.difference(Features.select_dtypes(include=['bool']).columns), 
        method=scaling_method
    )

    # correlation
    correlations = Features.corrwith(Labels[target_column]).abs().sort_values(ascending=False)
    # print(correlations[:20])

    Labels.reset_index(drop=True, inplace=True)
    Features.reset_index(drop=True, inplace=True)

    top20features = Features[correlations.index[:nofFeatures]]
    # print(top20features)


    X_train, X_test, Y_train, Y_test = train_test_split(top20features, Labels, test_size=0.2,random_state=42)

    return X_train, X_test, Y_train, Y_test