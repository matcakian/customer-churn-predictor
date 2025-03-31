# Customer Churn Prediction

A simple TensorFlow neural network that predicts customer churn using TensorFlow and scikit-learn.

## Scripts

- `prepare_dataset.py` – Loads and processes the Olist dataset into a structured NumPy array.
- `train_model.py` – Trains a binary classification model and saves the trained model and scaler.
- `client.py` – Loads the model and scaler to predict churn for a single client input.

## Output
- Saved model: `./models/model/`
- Saved scaler: `./scalers/scaler.pkl`
- Processed dataset: `./data/dataset.npy`