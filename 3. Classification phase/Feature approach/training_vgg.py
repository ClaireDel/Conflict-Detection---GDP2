from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import keras
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt



model = VGG16(weights="imagenet", include_top=False, input_shape= [64, 64, 3])
 
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


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


print('# of Samples:', len(y))
print('# of normal:', (y == 0).sum())
print('# of fight:', (y == 1).sum())


X = X / 255

y = tf.keras.utils.to_categorical(y, num_classes=None, dtype='float32')

history = CNN.fit(X, y, epochs=5, verbose=1, batch_size=64)


# learning curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Accuracy and loss')
plt.ylabel('Accuracy/Loss')
plt.xlabel('epoch')
plt.legend(['Accuracy', 'Loss'], loc='right')
plt.show()



# Save
CNN.save('model.h5')