# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:03:31 2025

@author: ilseh
"""

# Importing libraries.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def countplot(data, column_name, order, fig_size):
    """This function can be used to plot a seaborn count plot of a column in a pandas DataFrame. 
    
    data [pandas DataFrame]: data
    column_name [string]: name of the column whose data should be plotted
    order [string]: order (left to right) in which the bars should be plotted
    fig_size [tuple]: size of the plot; given as (width, height)
    """
    
    # Plotting normalized counts of the dataset column.
    plot = sns.countplot(data = data, x = column_name, hue = column_name,
                         order = order, hue_order = order,  palette = "deep",  
                         stat = "percent", legend = False)
    
    # Setting plot's title and axis labels. 
    plot.set_xlabel("Label")
    plot.set_ylabel("Percentage (%)") 
    plot.set_title(f"Label Distribution of ${column_name}$")
    
    # Modifying plot's size. 
    fig = plt.gcf()
    fig.set_size_inches(fig_size)
        
    plt.show()
    
def one_hot_encode(df, order = None):
    """This function returns a one-hot encoded version of a pandas Dataframe
    using the OneHotEncoder function from the sklearn library. 
    
    df [pandas DataFrame]: dataset
    order [list]: how binary labels should be assigned, e.g., ["Female", "Male"] to [0, 1]; 
                  can contain sublists for each of the binary columns in the dataset
    """
    
    # Label encoding the binary columns in the dataset.
    if order:
        for i in range(len(order)):
            labels = order[i]
            replaces = {labels[0]: 0, labels[1] : 1}
            df = df.replace(replaces)
    
    # Initializing one-hot encoder.
    one_hot_encoder = preprocessing.OneHotEncoder(sparse_output = False)
    one_hot_encoder.set_output(transform='pandas')
    cat_columns = df.select_dtypes(include=['object'])   # categorical columns
    num_columns = df.select_dtypes(exclude=['object'])   # numerical columns

    # Encoding the categorical columns and setting their column labels.
    cat_encoded = one_hot_encoder.fit_transform(cat_columns)
    
    # Concatenating the (encoded) categorical and numerical columns back into a single dataset. 
    df_encoded = pd.concat([cat_encoded.reset_index(drop = True), 
                            num_columns.reset_index(drop = True)], axis = 1)
    
    return df_encoded
    
def demographic_parity(df, s, y):
    """This function computes the demographic parity of the data with respect to the sensitive attribute.
    Note that the sensitive attribute should be encoded such that 1 equals the privileged group and 0 the 
    unprivileged group; in the target variable, 1 should equal the target variable's positive class.
    
    df [pandas DataFrame]: dataset
    s [string]: name of the sensitive attribute
    y [string]: name of the target variable
    """
    
    metric = ((df.loc[(df[s] == 1) & (df[y] == 1)].shape[0] / df.loc[df[s] == 1].shape[0]) - 
              (df.loc[(df[s] == 0) & (df[y] == 1)].shape[0] / df.loc[df[s] == 0].shape[0]))
        
    return metric

def disparate_impact(df, s, y):
    """This function computes the disparate impact of the data with respect to the sensitive attribute.
    Note that the sensitive attribute should be encoded such that 1 equals the privileged group and 0 the 
    unprivileged group; in the target variable, 1 should equal the target variable's positive class.
    
    df [pandas DataFrame]: dataset
    s [string]: name of the sensitive attribute
    y [string]: name of the target variable
    """
    
    metric = (((df.loc[(df[s] == 0) & (df[y] == 1)].shape[0] * df.loc[df[s] == 1].shape[0]) / 
               (df.loc[df[s] == 0].shape[0] * df.loc[(df[s] == 1) & (df[y] == 1)].shape[0])) - 0.8)
        
    return metric
    
def equal_opportunity(df, s, y_true, y_pred):
    """This function computes the equal opportunity of a classifier based on the true and predicted target values,
    with respect to the sensitive attribute. Note that the sensitive attribute should be encoded such that 1 equals 
    the privileged group and 0 the unprivileged group. The target variable should be binarily encoded too.
    
    df [pandas DataFrame]: dataset containing the true and predicted target class labels
    s [string]: column name of the sensitive attribute
    y_true [string]: column name of the true target values
    y_pred [string]: column name of the predicted target values
    """

    # Recall score for s = 1.
    recall_1 = recall_score(df[y_true].loc[df[s] == 1], df[y_pred].loc[df[s] == 1])
    
    # Recall score for s = 0.
    recall_0 = recall_score(df[y_true].loc[df[s] == 0], df[y_pred].loc[df[s] == 0])
    
    return recall_1 - recall_0
    
def utility_metrics(preds, y_true, conf_labels, conf_title):
    """This function provides a set of utility scores and a confusion matrix when given the true target values
    and a classifier's predicted target labels.
    
    preds [array-like]: array of target label predictions
    y_true [array-like]: array of true target labels
    conf_labels [list]: x- and y-axis tick labels of the confusion matrix, e.g., [0, 1]
    conf_title [string]: partial title for the confusion matrix plot 
    """
    
    # Accuracy score.
    accuracy = accuracy_score(y_true, preds)
    print("Accuracy on test data: {:.5f}%".format(accuracy*100))
    # Precision score.
    precision = precision_score(y_true, preds)
    print("Precision on test data: {:.5f}%".format(precision*100))
    # Recall score.
    recall = recall_score(y_true, preds)
    print("Recall on test data: {:.5f}%".format(recall*100))
    # AUROC score.
    auroc = roc_auc_score(y_true, preds)
    print(f"AUROC on test data: {auroc:.8f}")

    # Confusion matrix.
    conf_mat = confusion_matrix(y_true, preds)

    # Plotting the confusion matrix of the classifier.
    sns.heatmap(conf_mat, annot=True, cmap = "mako", fmt='g')
    plt.title(f"Confusion Matrix of {conf_title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks = [0.5, 1.5], 
               labels = conf_labels)   # setting ticks on x-axis to match target labels
    plt.yticks(ticks = [0.5, 1.5], 
               labels = conf_labels)   # setting ticks on y-axis to match target labels
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    