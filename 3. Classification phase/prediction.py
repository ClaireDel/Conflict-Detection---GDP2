import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import csv
from collections import defaultdict
import os
import cv2
import pickle
from sklearn.neural_network import MLPClassifier


# Load CSV files 

# Required step: store the keypoints of each image in a CSV file 
# On images already cropped: use cropping python file 
# use preprocessing python file as for the training data

data_test_normal = pd.read_csv('/Users/clair/Desktop/0OK.csv')
data_test_fight = pd.read_csv('/Users/clair/Desktop/1OK.csv')

columns = defaultdict(list)
with open('/Users/clair/Desktop/0OK.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    column_nums = range(len(headers))
    for row in reader:
        for i in column_nums:
            columns[headers[i]].append(row[i])               
columns = dict(columns)


columns = defaultdict(list)
with open('/Users/clair/Desktop/1OK.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    column_nums = range(len(headers))
    for row in reader:
        for i in column_nums:
            columns[headers[i]].append(row[i])               
columns = dict(columns)


# ..........; 


X_test_normal = data_test_normal.copy()
X_test_normal.drop(columns =['Filename'], inplace=True)
   
X_test_fight= data_test_fight.copy()
X_test_fight.drop(columns =['Filename'], inplace=True)

# ...........;

X_test_normal_array = np.array(X_test_normal)
y_test_normal = np.zeros(len(X_test_normal_array))
   
X_test_fight_array = np.array(X_test_fight)
y_test_fight = np.ones(len(X_test_fight_array))


# ...........;

X_test = list(X_test_normal_array) + list(X_test_fight_array)
X_test = np.array(X_test)
   
y_test = list(y_test_normal) + list(y_test_fight)
y_test = np.array(y_test)
   
print('# of Testing Samples:', len(y_test))
print('# of normal:', (y_test == 0).sum())
print('# of fight:', (y_test == 1).sum())

# ...........;

# Classifier
pickle_in = open("SVM.pickle", "rb") #NN
classifier = pickle.load(pickle_in)
   

# Prediction to evaluate accuracy
y_pred = classifier.predict(X_test)
print('Precision : ' + str(100*(f1_score(y_test, y_pred, average='micro'))) + " %")


# Confusion Matrix
plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues) 
plt.title('Confusion Matrix')
plt.show()
