import numpy as np
import os
import cv2
import pickle
import seaborn as sns


# Use cropping python file to obtain cropped images 
# with keypoints plotted on a black background

# Training
DIRECTORY_TRAIN = "/Users/clair/Desktop/train/" # with normal and fight

# Validation
DIRECTORY_TEST = "/Users/clair/Desktop/test/" # with normal and fight

CATEGORIES = ['normal', 'fight']
IMG_SIZE = 64


# Data
X = [] # train
X1 = [] # test

# Labels
y = [] # train
y1 = [] # test

def create_data():
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY_TRAIN, category)
        class_num_label = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                X.append(img_array)
                y.append(class_num_label)
            except Exception as e:
                pass
            
            
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY_TEST, category)
        class_num_label = CATEGORIES.index(category)
        for img in os.listdir(DIRECTORY_TEST):
            try:
                img_array = cv2.imread(os.path.join(DIRECTORY_TEST,img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                X1.append(img_array)
                y1.append(class_num_label)
            except Exception as e:
                pass

create_data()


# Reshaping 
# Train
SAMPLE_SIZE_TRAIN = len(y)
data = np.array(X).flatten().reshape(SAMPLE_SIZE_TRAIN, IMG_SIZE*IMG_SIZE, 3)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

# Test
SAMPLE_SIZE_TEST = len(y1)
data1 = np.array(X1).flatten().reshape(SAMPLE_SIZE_TEST, IMG_SIZE*IMG_SIZE, 3)
X1 = np.array(X1).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y1 = np.array(y1)


print("X shape : ", X.shape)
print("y shape : ", y.shape)
print("Data shape : ", data.shape)


names = ['normal', 'fight']
values = [(y == 0).sum(),(y == 1).sum()]
graph = sns.barplot(x=names, y=values, palette=['green','red'])
graph.set_title("Dataset repartition")
for k in range(1) :
    graph.text(k, 2500, values[k], fontsize=15, ha = 'center')


pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("X1.pickle", "wb")
pickle.dump(X1, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
pickle_out = open("y1.pickle", "wb")
pickle.dump(y1, pickle_out)
pickle_out.close()

pickle_out = open("data.pickle", "wb")
pickle.dump(data, pickle_out)
pickle_out.close()
pickle_out = open("data1.pickle", "wb")
pickle.dump(data1, pickle_out)
pickle_out.close()