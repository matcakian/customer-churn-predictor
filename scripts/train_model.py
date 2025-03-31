"""
This script trains and saves a binary classification model using TensorFlow.
It loads a NumPy dataset, splits the data, scales the features, and evaluates the model.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# Load the dataset
dataset = np.load("./data/dataset.npy", allow_pickle=True)
X, y = dataset[:, :4], dataset[:, 4]


# Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Build the model
model = tf.keras.Sequential([
	layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
	layers.Dense(64, activation="relu"),
	layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# Save the model
model.save("./models/model")


# Save the scaler
with open("./scalers/scaler.pkl", "wb") as f:
	pickle.dump(scaler, f)

