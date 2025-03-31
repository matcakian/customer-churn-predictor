"""
Loads a trained model and scaler to predict customer churn from client input.
"""

import tensorflow as tf
import numpy as np
import pickle


# Load the saved model and the scaler
model = tf.keras.models.load_model("./models/model")

with open("./scalers/scaler.pkl", "rb") as f:
	scaler = pickle.load(f)


# Get the client input; Example: [num_orders, avg_order_value, avg_review_score, total_reviews]
client_input = np.array([1, 10.25, 0, 10])

# Make a prediction
churn_prob = model.predict(scaler.transform(np.expand_dims(client_input, axis=0)))[0][0]

print(f"Customer churn probability: {churn_prob * 100:.2f}%")