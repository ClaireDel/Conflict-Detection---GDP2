import numpy as np
import cv2
import pickle
import tensorflow as tf
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CNN = tf.keras.models.load_model('model.h5')

# Testing data
pickle_in = open("X1.pickle", "rb")
X1 = pickle.load(pickle_in)
pickle_in = open("y1.pickle", "rb")
y1 = pickle.load(pickle_in)
pickle_in = open("data1.pickle", "rb")
data1 = pickle.load(pickle_in)



# Normalization
X1 = X1 / 255


# Transformation
y1 = tf.keras.utils.to_categorical(y1, num_classes=None, dtype='int64')


# Prediction
y_pred = CNN.predict(X1)
y_pred1 = np.argmax(y_pred,axis=1)
y2 = np.argmax(y1,axis=1)


# Count of errors
E = []
e = 0
for k in range(len(y1)) :
    E.append(y2[k]-y_pred1[k])
for k in range(len(E)) :
    if E[k] != 0 : 
        e += 1
        
print("Prediction :")
print("Error count : ", e, "on", len(y1), ' ', "(" + str(round(e/len(E)*100, 3)) + "%" + ")")


# Confusion matrix
cm = sklearn.metrics.confusion_matrix(y2, y_pred1)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', square=True, linewidths=.5, cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion matrix')
plt.show()

    



  