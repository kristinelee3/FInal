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

def evaluate(knn_model,GNB_model,model,X_train,X_test, X_val,y_train, y_test, y_val,features,depth_limit):
    #knn
    y_pred_knn = knn_model.predict(X_val)

    conf = confusion_matrix(y_val,y_pred_knn)
    print(conf)
    TP=conf[1][1]
    TN = conf[0][0]
    FN=conf[1][0]
    FP=conf[0][1]
    sns.heatmap(conf, annot=True, fmt='d', )

    accuracy = accuracy_score(y_val,y_pred_knn)
    precision = precision_score(y_val,y_pred_knn)
    recall = recall_score(y_val,y_pred_knn)
    specificity = FP/(TN+FP)
    f1 = f1_score(y_val,y_pred_knn)
    print(accuracy)
    print(precision)
    print(recall)
    print(specificity)
    print(f1)

    #GaussianNB
    y_pred_GNB = knn_model.predict(X_val)

    conf = confusion_matrix(y_val,y_pred_GNB)
    print(conf)
    TP=conf[1][1]
    TN = conf[0][0]
    FN=conf[1][0]
    FP=conf[0][1]
    sns.heatmap(conf, annot=True, fmt='d', )

    accuracy = accuracy_score(y_val,y_pred_GNB)
    precision = precision_score(y_val,y_pred_GNB)
    recall = recall_score(y_val,y_pred_GNB)
    specificity = FP/(TN+FP)
    f1 = f1_score(y_val,y_pred_GNB)
    print(accuracy)
    print(precision)
    print(recall)
    print(specificity)
    print(f1)

    #DecisionTree
    y_pred_train = model.predict(X_train[features])
    y_pred_test = model.predict(X_val[features])
    train_accuracy = metrics.accuracy_score(y_train,y_pred_train)
    train_f1 = metrics.f1_score(y_train,y_pred_train)
    test_accuracy = metrics.accuracy_score(y_val,y_pred_test)
    test_f1 = metrics.f1_score(y_val,y_pred_test)

    print('---Training Performance')
    print('Train Accuracy:', train_accuracy)
    print('Train f1 Score:',train_f1)
    print('---Testing Performance')
    print('Test Accuracy:', test_accuracy)
    print('Test f1 Score:',test_f1)

    plt.figure(figsize=(4,4))
    plot_tree(model, feature_names=features,class_names=['precip','no_precip'],filled=True)
    plt.title(f"Decision Tree (Features: {features}, Max Depth: {depth_limit})")
    plt.show()

    conf = confusion_matrix(y_val,y_pred_test)
    print(conf)
    TP=conf[1][1]
    TN = conf[0][0]
    FN=conf[1][0]
    FP=conf[0][1]
    sns.heatmap(conf, annot=True, fmt='d', )

    accuracy = accuracy_score(y_val,y_pred_test)
    precision = precision_score(y_val,y_pred_test)
    recall = recall_score(y_val,y_pred_test)
    specificity = FP/(TN+FP)
    f1 = f1_score(y_val,y_pred_test)
    print(accuracy)
    print(precision)
    print(recall)
    print(specificity)
    print(f1)


    knn_model = KNeighborsClassifier(n_neighbors=21)
    knn_model.fit(X_train,y_train)
    y_pred_knn = knn_model.predict_proba(X_val)[:, 1]
    print(roc_auc_score(y_val, y_pred_knn))

    #GaussianNB
    knn_model = GaussianNB()
    knn_model.fit(X_train,y_train)
    y_pred_GNB = knn_model.predict_proba(X_val)[:, 1]
    print(roc_auc_score(y_val, y_pred_GNB))

    #DecisionTree
    features = ['temp', 'dew', 'humidity','windgust']#,'precipprob']
    depth_limit = 4
    model = DecisionTreeClassifier(criterion='entropy',max_depth=depth_limit)
    model.fit(X_train[features],y_train)
    y_pred_test = model.predict_proba(X_val[features])[:, 1]
    print(roc_auc_score(y_val, y_pred_test))
    return y_pred_GNB
