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

data_normal= pd.read_csv('/Users/clair/Desktop/Normal.csv')
data_fight = pd.read_csv('/Users/clair/Desktop/Fight.csv')
data_test = pd.read_csv('/Users/clair/Desktop/test.csv')


columns = defaultdict(list)
with open('/Users/clair/Desktop/Normal.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    column_nums = range(len(headers)) # Do NOT change to xrange
    for row in reader:
        for i in column_nums:
            columns[headers[i]].append(row[i])
# Following line is only necessary if you want a key error for invalid column names
columns = dict(columns)



columns = defaultdict(list)
with open('/Users/clair/Desktop/Fight.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    column_nums = range(len(headers)) # Do NOT change to xrange
    for row in reader:
        for i in column_nums:
            columns[headers[i]].append(row[i])
# Following line is only necessary if you want a key error for invalid column names
columns = dict(columns)



columns = defaultdict(list)
with open('/Users/clair/Desktop/test.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    column_nums = range(len(headers)) # Do NOT change to xrange
    for row in reader:
        for i in column_nums:
            columns[headers[i]].append(row[i])
# Following line is only necessary if you want a key error for invalid column names
columns = dict(columns)



# ..........; 

X_normal = data_normal.copy()
X_normal.drop(columns =['Filename'], inplace=True)

X_fight= data_fight.copy()
X_fight.drop(columns =['Filename'], inplace=True)

X_test= data_test.copy()
X_test.drop(columns =['Filename'], inplace=True)


# ...........;

X_normal_array = np.array(X_normal)
y_normal = np.zeros(len(X_normal_array))

X_fight_array = np.array(X_fight)
y_fight = np.ones(len(X_fight_array))

X_test_array = np.array(X_test)
y_test = np.ones(len(X_test_array))

# ...........;

X = list(X_normal_array) + list(X_fight_array)
X = np.array(X)

y = list(y_normal) + list(y_fight)
y = np.array(y)

# Unseen data
X_val = list(X_test_array) 
X_val = np.array(X_val)
y_val = np.array(y_test)

print('# of Training Samples:', len(y))
print('# of normal:', (y == 0).sum())
print('# of fight:', (y == 1).sum())
print('# of Testing Samples:', len(y_val))

# ................;

# Training
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# Classifier
clf = SVC(random_state=0)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, Y_train)


# Prediction to evaluate accuracy
y_pred = clf.predict(X_test)
print('Précision : ' + str(100*(f1_score(Y_test, y_pred, average='micro'))) + " %")


# Confusion Matrix
plot_confusion_matrix(clf, X_test, Y_test, cmap=plt.cm.Blues) 
plt.title('Confusion Matrix')
plt.show()


# Prediction on unseen data (Validation)
y_pred_val = clf.predict(X_val)


# Save the model
pickle_out = open("SVM.pickle", "wb")
# pickle_out = open("MLP.pickle", "wb")
pickle.dump(clf, pickle_out)
pickle_out.close()



