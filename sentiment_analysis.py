import os
import glob
import pandas as pd
from src.preprocess import prepare_data, vectorize_data
from src.models import train_and_evaluate

# Load and combine split CSV files
data_folder = os.path.join("data")
file_list = glob.glob(os.path.join(data_folder, "ratings_Electronics_part_*.csv"))

if not file_list:
    raise FileNotFoundError("No ratings_part_*.csv files found in 'data' folder.")

# Combine all parts into one DataFrame
df_list = [pd.read_csv(file) for file in file_list]
combined_df = pd.concat(df_list, ignore_index=True)

print("Using split CSV files:", file_list)

df = pd.read_csv("data/ratings_Electronics.csv")
print(df.head())


# Prepare the data
df, label_encoder = prepare_data(combined_df)

# Vectorize the data
(X_train, X_test, y_train, y_test), tfidf = vectorize_data(df)

# Train and evaluate models
train_and_evaluate(X_train, X_test, y_train, y_test)
