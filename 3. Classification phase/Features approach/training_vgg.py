from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import keras
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import plot_confusion_matrix

# Load data
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

print('# of Samples:', len(y))
print('# of normal:', (y == 0).sum())
print('# of fight:', (y == 1).sum())



# Normalization and train test split
X = X / 255
y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=42)




# Model
model = VGG16(weights="imagenet", include_top=False, input_shape= [64, 64, 3])
# print(model.summary())

for layer in model.layers:
   layer.trainable = False

output_vgg16_conv = model.output

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)
x = Dense(2, activation='softmax')(x) 

CNN = keras.models.Model(inputs=model.input, outputs=x)
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(CNN.summary())
history = CNN.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=9, verbose=1, batch_size=64)


y_pred = CNN.predict(X_test)

# Save
CNN.save('model_vgg.h5')



# Evaluation
epochs=np.linspace(0,9,9)


# # Accuracy and loss
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['loss'])
# plt.title('Accuracy and loss')
# plt.ylabel('Accuracy/Loss')
# plt.xlabel('epoch')
# plt.legend(['Accuracy', 'Loss'], loc='right')
# plt.show()

# Plot training history
plt.plot(history.history['loss'], label='training')
plt.title('Training curve (loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot training history
plt.plot(history.history['accuracy'], label='training')
plt.title('Training curve (accuracy)')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Tr and Val loss curves
plt.plot(epochs, history.history['loss'], 'r')
plt.plot(epochs, history.history['val_loss'], 'b')
plt.ylim((0,0.5))
plt.title('Training and validation loss curves')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Tr and Val accuracy curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.title('Training and validation accuracy curves')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

