import torch
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,confusion_matrix,roc_curve,auc,roc_auc_score,matthews_corrcoef
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split,KFold

# CLS data reading function
def readClsData(cls_datafile,cls_data):
    cnt=0
    with open(cls_datafile,"r") as f:
        d=f.readline()
        while d:
            d=f.readline().split() 
            if d:
                temp=[]
                for i in d:
                    i=float(i)
                    temp.append(i)
                cnt+=1
                cls_data.append(temp)
            f.readline()

# Model test function
def modelPerformance(y_true, y_pred):
    print('-------------------------------------------------------------')
    fpr, _, _ = roc_curve(y_pred, y_true)
    print('ACC: ' + str(accuracy_score(y_true, y_pred)))
    print('Precision: ' + str(precision_score(y_true, y_pred)))
    print('Sn: ' + str(recall_score(y_true, y_pred)))
    print('Sp: ' + str(1 - fpr[1]))
    print("F1: " + str(f1_score(y_true, y_pred)))
    print("Kappa: " + str(cohen_kappa_score(y_true, y_pred)))
    print("MCC: " + str(matthews_corrcoef(y_true, y_pred)))
    print('-------------------------------------------------------------')

# CLS data acquisition
negative_train_cls=[]
negative_test_cls=[]
positive_train_cls=[]
positive_test_cls=[]
path_ne_train_cls="clsdata/negative_train_cls.txt"
path_ne_test_cls="clsdata/negative_test_cls.txt"
path_po_train_cls="clsdata/positive_train_cls.txt"
path_po_test_cls="clsdata/positive_test_cls.txt"

# PSSM data acquisition
kp_ne_test = np.loadtxt(open("pssmdata/kp_ne_test.csv","rb"),delimiter=",",skiprows=1)
kp_ne_train = np.loadtxt(open("pssmdata/kp_ne_train.csv","rb"),delimiter=",",skiprows=1)
kp_po_test = np.loadtxt(open("pssmdata/kp_po_test.csv","rb"),delimiter=",",skiprows=1)
kp_po_train = np.loadtxt(open("pssmdata/kp_po_train.csv","rb"),delimiter=",",skiprows=1)

dpc_ne_test = np.loadtxt(open("pssmdata/dpc_ne_test.csv","rb"),delimiter=",",skiprows=1)
dpc_ne_train = np.loadtxt(open("pssmdata/dpc_ne_train.csv","rb"),delimiter=",",skiprows=1)
dpc_po_test = np.loadtxt(open("pssmdata/dpc_po_test.csv","rb"),delimiter=",",skiprows=1)
dpc_po_train = np.loadtxt(open("pssmdata/dpc_po_train.csv","rb"),delimiter=",",skiprows=1)

sf_ne_test = np.loadtxt(open("pssmdata/sf_ne_test.csv","rb"),delimiter=",",skiprows=1)
sf_ne_train = np.loadtxt(open("pssmdata/sf_ne_train.csv","rb"),delimiter=",",skiprows=1)
sf_po_test = np.loadtxt(open("pssmdata/sf_po_test.csv","rb"),delimiter=",",skiprows=1)
sf_po_train = np.loadtxt(open("pssmdata/sf_po_train.csv","rb"),delimiter=",",skiprows=1)

# Generate training set and test set (without PSSM integration)
a=np.array(positive_train_cls)
b=np.array(negative_train_cls)
x_train=np.vstack([a,b])
a=np.array(positive_test_cls)
b=np.array(negative_test_cls)
x_test=np.vstack([a,b])
y_train=np.hstack([np.ones(1005), np.zeros(1059)])
y_test=np.hstack([np.ones(218), np.zeros(260)])

# Feature selection 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1,max_depth=3)
forest.fit(x_train, y_train)
importances = forest.feature_importances_

# Subscript sorting
indices = np.argsort(importances)[::-1] 

x_train_select=[]
x_test_select=[]

# Dimensional parameters (adjustable)
K=1500

for i in x_train:
  traint=[]
  for k in indices[0:K]:
    traint.append(i[k])
  x_train_select.append(traint)
for i in x_test:
  testt=[]
  for k in indices[0:K]:
    testt.append(i[k])
  x_test_select.append(testt)
x_train_select=np.array(x_train_select)
x_test_select=np.array(x_test_select)

# Enable these two lines of code only after feature selection
x_train=x_train_select 
x_test=x_test_select 

# Model testing
smodel5 = svm.SVC(gamma='scale', 
          C=10, 
          decision_function_shape='ovr', 
          kernel='rbf')
smodel5.fit(x_train,y_train)
result=smodel5.predict(x_test)
modelPerformance(y_test,result)


forest1 = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1,max_depth=3)
forest1.fit(x_train, y_train)
result=forest1.predict(x_test)
modelPerformance(y_test,result)


NB1 = GaussianNB()
NB1.fit(x_train, y_train)
result=NB1.predict(x_test)
modelPerformance(y_test,result)


XGB1 = XGBClassifier(max_depth=15,
          learning_rate=0.1,
          n_estimators=2000,
          min_child_weight=5,
          max_delta_step=0,
          subsample=0.8,
          colsample_bytree=0.7,
          reg_alpha=0,
          reg_lambda=0.4,
          scale_pos_weight=0.8,
          objective='binary:logistic',
          eval_metric='auc',
          seed=1440,
          gamma=0)
XGB1.fit(x_train, y_train)
y_pred = XGB1.predict(x_test)
result = [round(value) for value in y_pred]
modelPerformance(y_test,result)


knc1 = KNN(n_neighbors =5)
knc1.fit(x_train,y_train)
result = knc1.predict(x_test)
modelPerformance(y_test,result)
