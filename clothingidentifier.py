import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# keras datasets will automatically load in training and testing sets, no need to manually split
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# 10 label numbers correspond to 10 different class items
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5) # epochs refer to how many times the model will see the SAME image

prediction = model.predict(test_images)

# setting up a simple for loop to print and return the actual image and the predicted class
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
# prediction will return in an array of 10 values - each representing the probabilities of being the certain class
# argmax will return the largest value i.e. the predicted class