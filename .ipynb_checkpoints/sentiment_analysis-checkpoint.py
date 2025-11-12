# sentiment_analysis.py

import os
from src.preprocess import prepare_data, vectorize_data
from src.models import train_and_evaluate

# Use the correct filename
data_path = os.path.join("data", "ratings.csv")

# Check if file exists
if not os.path.isfile(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

print("Using local CSV file:", data_path)

# Prepare the data
df, label_encoder = prepare_data(data_path)

# Vectorize the data
(X_train, X_test, y_train, y_test), tfidf = vectorize_data(df)

# Train and evaluate models
train_and_evaluate(X_train, X_test, y_train, y_test)
