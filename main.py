from src.data import load_data
from src.features import build_features
from src.models import evaluate_model,train_model
from src.visualization import visualize
from tests import tests

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

data=load_data.load()
data=build_features.feature(data)
knn_model, GNB_model, model,X_train,X_test, X_val,y_train, y_test,y_val,features,depth_limit = train_model.train_model(data)

y_pred_GNB=evaluate_model.evaluate(knn_model,GNB_model,model,X_train,X_test, X_val,y_train, y_test, y_val,features,depth_limit)
visualize.plot(data)

tests.test(knn_model,X_train,y_train,y_pred_GNB,y_test,X_test)
