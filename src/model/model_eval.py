# Necessary libraries
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test dataset
# test_data = pd.read_csv("./data/processed/test_processed.csv")
def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Separate features and target variable and convert to numpy arrays
# x_test = test_data.iloc[:, 0:-1].values
# y_test = test_data.iloc[:, -1].values
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        x = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return x, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

# Load the trained model from file
# model = pickle.load(open("model.pkl", "rb"))
def load_model(filepath: str):
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

# Model evaluation
# Make predictions on the test set
# y_pred = model.predict(x_test)

# Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# Store the evaluation metrics in a dictionary (json file)
# metrics = {
#     "accuracy": accuracy,
#     "precision": precision,
#     "recall": recall,
#     "f1_score": f1
# }

def evaluation_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        y_pred = model.predict(x_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")
    
# Save the metrics to a JSON file
# with open("metrics.json", "w") as file:
#     json.dump(metrics, file, indent=4)

def save_metrics(metrics_dict: dict, filepath: str) -> None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}")
    
# Main function to execute the steps
def main():
    try:
        # Paths
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        # Load test dataset
        test_data = load_data(test_data_path)

        # Prepare data
        x_test, y_test = prepare_data(test_data)

        # Load the trained model from file
        model = load_model(model_path)

        # Model evaluation
        metrics = evaluation_model(model, x_test, y_test)

        # Save the metrics to a JSON file
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()