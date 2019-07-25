# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:33:05 2019

@author: Spryzen
"""
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

data.info()
#loading of data
data=pandas.read_csv(r"file:///C:/Users/tanma/OneDrive/Desktop/trial.csv")
cor=data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()
# 10 12 midsem1 midsem2 medsem3 midsem4 midsem5 sem1 sem2 sem3 sem4 
plt.figure(figsize=(12,5))
sns.distplot(data.ten[data.Gender=='MALE'])
sns.distplot(data.ten[data.Gender=='FEMALE'])
plt.legend(['M','F'])
plt.show()

plt.figure(figsize=(12,5))
sns.distplot(data.twelve[data.Gender=='MALE'])
sns.distplot(data.twelve[data.Gender=='FEMALE'])
plt.legend(['M','F'])
plt.show()


data.info()  #object(categorical value)

data2.shape
data.columns
cat=['Gender', 'ten', 'twelve', 'midsem1', 'midsem2', 'midsem3',
       'midsem4', 'midsem5', 'sem1', 'sem2', 'sem3','attendance1',
       'attendance2', 'attendance3', 'attendance4', 'attendance5', 'active',
       'feed']
len(cat)

for i in cat:
    c=list(data[i].unique())
    plt.figure(figsize=(12,5))
    for j in c:
        sns.distplot(data.sem4[data[i]==j])
    plt.title(i)
    plt.legend(c)
    plt.show()
    
    data2=data[[ 'Gender', 'ten', 'twelve', 'midsem1', 'midsem2', 'midsem3',
       'midsem4', 'midsem5', 'sem1', 'sem2', 'sem3', 'sem4', 'attendance1',
       'attendance2', 'attendance3', 'attendance4', 'attendance5', 'active',
       'feed']]
    
xdata=data2.drop(['sem4'],axis=1)
ydata=data2['sem4']


from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
xdata.Gender=le1.fit_transform(xdata.Gender) # it is converted male as 1 and female as 1

le2=LabelEncoder()
xdata.active=le2.fit_transform(xdata.active)  #it is converted urban as 1 and rural as 0

le3=LabelEncoder()
xdata.feed=le3.fit_transform(xdata.feed) # school support yes as 1 and no as 0

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(xdata,ydata,test_size=0.2,random_state=3)


### Decision Tree
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(max_depth=2)
#train the algorithm
model.fit(xtr,ytr)

#check the r2 score of the model(accuracy)
model.score(xtr,ytr)#train data

model.score(xts,yts)#test data

"""
gender
ten
twelve
mid1 
mid2
mid3
mid4
mid5
sem1
sem2
sem3
attendance1
attendance2
attendance3
attendance4
attendance5
active
feed
"""

ip=[[1,85,68,76,65,87,78,67,98,87,98,87,80,65,98,88,1,0]]
model.predict(ip)

