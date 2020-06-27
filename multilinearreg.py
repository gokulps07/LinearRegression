# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:35:27 2020

@author: PSG
"""

import numpy as np
import matplotlib.pyplot as mpl
import pandas as  pa
#importing csv file
data=pa.read_csv('insurance.csv')
#Seperating Dependent and Independent Variables
X=data.iloc[:,:-1].values
y=data.iloc[:,6].values
#Encoding for changing string values to Integer Using OneHotEncoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
laben=LabelEncoder()
X[:,1]=laben.fit_transform(X[:,1])
one=OneHotEncoder(categorical_features=[1])
laben1=LabelEncoder()
X[:,4]=laben.fit_transform(X[:,4])
one1=OneHotEncoder(categorical_features=[4])
laben1=LabelEncoder()
X[:,5]=laben.fit_transform(X[:,5])
one2=OneHotEncoder(categorical_features=[5])
X=one.fit_transform(X).toarray()
X=X[:,1:]
X=one1.fit_transform(X).toarray()
X=X[:,1:]
X=one2.fit_transform(X).toarray()
X=X[:,1:]
#Seperating of train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.001,random_state=0)
#creation of linear regression model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
yres=lr.predict(x_test)
'''
#Plotting of Regression Model
mpl.plot(lr.predict(x_train))
mpl.title('Regression Model')

mpl.show()
'''








