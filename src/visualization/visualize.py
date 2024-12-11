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


def plot(data):
    # Visualize the used pin number distribution
    sns.countplot(x='is_precip', data=data)
    plt.title('Precipitation Distribution')
    plt.show()


    # Create a scatter plot to show the breakdown of fraudulent and non-fraudulent transactions
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=data,
        x='visibility',
        y='solarradiation',
        hue='is_precip',
        palette={0: 'green', 1: 'red'},
    )
    #plt.title('Fraudulent vs Non-Fraudulent Transactions: Distance from Home vs Ratio to Median Purchase Price')
    #plt.xlabel('Distance from Home')
    #plt.ylabel('Ratio to Median Purchase Price')
    plt.show()

    # Create a scatter plot to show the breakdown of fraudulent and non-fraudulent transactions
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=data,
        x='windspeed',
        y='sealevelpressure',
        hue='is_precip',
        palette={0: 'green', 1: 'red'},
    )
    #plt.title('Fraudulent vs Non-Fraudulent Transactions: Distance from Home vs Ratio to Median Purchase Price')
    #plt.xlabel('Distance from Home')
    #plt.ylabel('Ratio to Median Purchase Price')
    plt.show()

    # Create a scatter plot to show the breakdown of fraudulent and non-fraudulent transactions
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=data,
        x='tempmax',
        y='winddir',
        hue='is_precip',
        palette={0: 'green', 1: 'red'},
    )
    #plt.title('Fraudulent vs Non-Fraudulent Transactions: Distance from Home vs Ratio to Median Purchase Price')
    #plt.xlabel('Distance from Home')
    #plt.ylabel('Ratio to Median Purchase Price')
    plt.show()

    #sns.pairplot(data, hue="is_precip")

    
