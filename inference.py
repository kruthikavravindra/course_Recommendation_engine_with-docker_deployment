#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def inference():

    # Load the trained model and encoding scheme
    with open('model.pkl', 'rb') as f:
        knn = pickle.load(f)

    with open('encoding.pkl', 'rb') as f:
        ct = pickle.load(f)

    # Load the new data from a CSV file
    test = "./test.csv"
    data= pd.read_csv(test)
    new_data = pd.read_csv('test.csv')

    # Apply the encoding scheme to the new data
    X_new = ct.transform(new_data)

    # Make predictions using the trained model
    y_pred = knn.predict(X_new)
    y_pred=pd.DataFrame(y_pred,columns=['Recommended_course'])
    new_data=pd.concat([new_data,y_pred],axis=1)
    print(new_data)
    
if __name__ == '__main__':
    inference()


# In[ ]:




