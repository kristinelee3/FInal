import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score

def train_model(data):
    X = data[['temp', 'dew', 'humidity','windgust']]

    y = data['is_precip']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=2/3, random_state=42)

    #Knn
    knn_model = KNeighborsClassifier(n_neighbors=3)#21)
    knn_model.fit(X_train,y_train)

    max_acc=0
    neigh=0
    max_f1=0
    neighhh=0
    for i in range(1,51):
        if i ==5:
            continue
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(X_train,y_train)
        y_pred = knn_model.predict(X_val)
        acc=accuracy_score(y_val,y_pred)
        f1 = f1_score(y_val,y_pred)
        if acc>max_acc:
            max_acc=acc
            neigh=i
        if f1>max_f1:
            max_f1=f1
            neighhh=i
    print(max_acc,neigh,max_f1,neighhh)

    #GaussianNB
    GNB_model = GaussianNB()
    GNB_model.fit(X_train,y_train)

    #DecisionTree
    features = ['temp', 'dew', 'humidity','windgust']#,'precipprob']
    depth_limit = 4
    model = DecisionTreeClassifier(criterion='entropy',max_depth=depth_limit)
    model.fit(X_train[features],y_train)
    return knn_model, GNB_model, model,X_train,X_test, X_val,y_train, y_test, y_val,features,depth_limit
