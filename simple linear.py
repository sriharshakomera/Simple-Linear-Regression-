# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:56:31 2019

@author: Sriharsha Komera
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset= pd.read_csv("F:/Krish/Simple linear/Salary.csv")

# divide dataset into x and y
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# splitting the data based on train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
SLR=LinearRegression()
SLR.fit(X_train,y_train)
y_predict=SLR.predict(X_test)

#implementation of the graph

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,SLR.predict(X_train))
plt.show()

from sklearn.metrics import r2_score
r2_score(y_test,y_predict)