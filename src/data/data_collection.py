# Necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

# Load parameters from params.yaml ( we are providing type hints here)
# test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]  # Load the YAML file
def load_params(filepath : str) -> float:
    # Exception handler
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")

# Load dataset
# data_url = r"https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv"
# df = pd.read_csv(data_url)
def load_data(filepath : str) -> pd.DataFrame:
    # Exception handler
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Split dataset into training and testing sets
# train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Exception handler
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error splitting data: {e}")

# Define data path (folders)
# data_path = os.path.join("data", "raw")
# os.makedirs(data_path)

# Save train and test sets to CSV files
# train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
def save_data(data: pd.DataFrame, filepath: str) -> None:
    # Exception handler
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

# Main function to execute the steps
def main():
    # Exception handler
    try:
        # Data and parameters paths
        data_filepath = r"https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv"
        params_filepath = "params.yaml"
        raw_data_path = os.path.join("data", "raw")

        # Load dataset and parameters
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)

        # Split dataset into training and testing sets
        train_data, test_data = split_data(data, test_size)

        # Save train and test sets to CSV files
        os.makedirs(raw_data_path)
        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()