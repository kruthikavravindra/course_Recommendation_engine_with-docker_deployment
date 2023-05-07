#!/usr/bin/env python
# coding: utf-8

# In[3]:


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import pickle

def train():

    # Load, read and normalize training data
    training = "./train.csv"
    data= pd.read_csv(training)
        
    # Separate the features and the target variable
    X = data.drop('course', axis=1)
    y = data['course']

    # Define the categorical columns and apply one-hot encoding
    cat_cols = ['gender', 'stream', 'subject']
    ct = ColumnTransformer([('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
    X = ct.fit_transform(X)

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the k-NN classifier
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the k-NN classifier
    knn.fit(X_train, y_train)

    # Save the trained model and encoding scheme
    with open('model.pkl', 'wb') as f:
        pickle.dump(knn, f)

    with open('encoding.pkl', 'wb') as f:
        pickle.dump(ct, f)
        
if __name__ == '__main__':
    train()


# In[ ]:




