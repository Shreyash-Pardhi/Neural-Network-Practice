# Expt.a)
# Implementing handwritten character recognition using shallow network in Keras
# **Loading the MNIST dataset in Keras**
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

# plot the sample mnist digits
import numpy as np
indexes = np.random.randint(0, train_images.shape[0], size=25)
images = train_images[indexes]
labels = train_labels[indexes]
print(labels)
plt.figure(figsize=(4,4))
plt.suptitle("MNIST Sample Images")
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
plt.show(block=False)
plt.savefig("mnist-samples.png")


# displaying shape of training and testing data
print(train_images.shape)
print(test_images.shape)

# **The network architecture**
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(30, activation="relu"),
    layers.Dense(10, activation="softmax")])

# model compilation
# If you have two or more classes and  the labels are integers, the SparseCategoricalCrossentropy should be used.

model.compile(optimizer="Adamax",
                loss= "sparse_categorical_crossentropy",
                metrics=["acc"])
'''
model.compile(optimizer="rmsprop",
                loss= "sparse_categorical_crossentropy",
                metrics=["acc"])
'''

#preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

#fitting the model
history = model.fit(train_images, train_labels, validation_split=0.33, epochs=50, batch_size=32)

#Printing model summary
print(model.summary())

#**Evaluating the model on new data**
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
print(f"test loss: {test_loss}")

#**Using the trained model to make predictions for few samples**
test_digits = test_images[0:10]
test_desired = test_labels[0:10]
predictions = model.predict(test_digits)
print('Desired output Model Output')
for i in range(len(test_digits)):
    print(test_desired[i],"   ",predictions[i].argmax())

# print(history.history.keys())

# Visualization of Model performance
plt.figure(figsize=(12,7))
plt.suptitle("Model performance")
#  displaying loss curves
plt.subplot(1, 2, 1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Training and Validation Loss')
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()

# displaying accuarcy curves
plt.subplot(1, 2, 2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training and Validation Accuracy')
plt.plot(history.history['acc'], label="training accuracy")
plt.plot(history.history['val_acc'], label="validation accuracy")
plt.legend()
plt.show()

# Model performance obtained using 50 epochs
# (Training accuracy=98.28, validation accuracy =96.13, test accuracy of 96.42, training loss =0.0625, validation loss=0.1386, test loss =0.124)