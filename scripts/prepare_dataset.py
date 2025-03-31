"""
This script loads and merges Olist order, customer, payment, and review datasets,
then computes per-customer features and exports the result as a NumPy array.
"""

import pandas as pd
import numpy as np


# Import the datasets
orders = pd.read_csv("./archive/olist_orders_dataset.csv")
customers = pd.read_csv("./archive/olist_customers_dataset.csv")
payments = pd.read_csv("./archive/olist_order_payments_dataset.csv")
reviews = pd.read_csv("./archive/olist_order_reviews_dataset.csv")


# Merge the datasets
order_payments = payments.groupby("order_id")["payment_value"].sum().reset_index()
orders_with_payments = orders.merge(order_payments, on="order_id", how="inner")

order_reviews = reviews.groupby("order_id").agg({
	"review_score": ["mean", "count"]
}).reset_index()

order_reviews.columns = ("order_id", "avg_review_score", "num_reviews")
orders_with_reviews = orders_with_payments.merge(order_reviews, on="order_id", how="inner")

orders_with_customers = orders_with_reviews.merge(customers, 
	on="customer_id", how="inner")


# Create the features
orders_with_customers["order_purchase_timestamp"] = pd.to_datetime(
	orders_with_customers["order_purchase_timestamp"])

customer_metrics = orders_with_customers.groupby("customer_unique_id").agg({
	"order_id": "count",
	"payment_value": "mean",
	"avg_review_score": "mean",
	"num_reviews": "sum",
	"order_purchase_timestamp": "max"
})

customer_metrics.columns = ("num_orders", "avg_order_value", 
	"avg_review_score", "total_reviews", "most_recent_purchase")


# Create the labels
CURRENT_TIME = customer_metrics["most_recent_purchase"].max()
customer_metrics["days_since_last_purchase"] = (
	 CURRENT_TIME - customer_metrics["most_recent_purchase"]).dt.days

customer_metrics["churned"] = (customer_metrics["days_since_last_purchase"] > 365).astype(int)
customer_metrics.drop(["most_recent_purchase", "days_since_last_purchase"], axis=1, inplace=True)


# Save the dataset
np.save("./data/dataset.npy", customer_metrics.to_numpy())

