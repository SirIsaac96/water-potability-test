# Necessary libraries
import pandas as pd
import numpy as np
import os

# Load train and test datasets
# train_df = pd.read_csv("./data/raw/train.csv")
# test_df = pd.read_csv("./data/raw/test.csv")
def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Function to handle missing values by replacing them with the median of the column
def handle_missing_values(data):
    try:
        for column in data.columns:
            if data[column].isnull().any():
                median_value = data[column].median()
                data[column].fillna(median_value, inplace=True)
        return data
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")

# Handle missing values in both train and test datasets
# train_processed_data = handle_missing_values(train_df)
# test_processed_data = handle_missing_values(test_df)

# Save the processed datasets to new CSV files
# processed_data_path = os.path.join("data", "processed")
# os.makedirs(processed_data_path, exist_ok=True)
def save_data(data: pd.DataFrame, filepath: str) -> None:
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")    

# train_processed_data.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
# test_processed_data.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)

# Main function to execute the steps
def main():
    try:
        # Raw and processed data paths
        raw_data_path = "./data/raw/"
        processed_data_path = "./data/processed/"

        # Load train and test datasets
        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        # Handle missing values in both train and test datasets
        train_processed_data = handle_missing_values(train_data)
        test_processed_data = handle_missing_values(test_data)

        # Save the processed datasets to new CSV files
        os.makedirs(processed_data_path)
        save_data(train_processed_data, os.path.join(processed_data_path, "train_processed.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed.csv"))
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()