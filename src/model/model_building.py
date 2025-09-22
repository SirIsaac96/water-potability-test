import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

# Load model parameters from params.yaml
# n_estimators = yaml.safe_load(open("params.yaml", "r"))["model_building"]["n_estimators"]
def load_params(filepath : str) -> int:
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")

# Load processed training data
# train_data = pd.read_csv("./data/processed/train_processed.csv")
def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

# Separate features and target variable and convert to numpy arrays
# x_train = train_data.iloc[:, 0:-1].values
# y_train = train_data.iloc[:, -1].values
# x_train = train_data.drop(columns=["Potability"], axis=1)
# y_train = train_data["Potability"]
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        x = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return x, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

# Initialize and train the RandomForestClassifier
# model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
# model.fit(x_train, y_train)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(x_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

# Save the trained model to a file using pickle
# pickle.dump(model, open("model.pkl", "wb"))
def save_model(model: RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {filepath}: {e}")
    
# Main function to execute the steps
def main():
    try:
        # Paths
        params_path = "params.yaml"
        processed_data_path = "./data/processed/train_processed.csv"
        model_name = "models/model.pkl"

        # Load parameters and processed training data
        n_estimators = load_params(params_path)
        train_data = load_data(processed_data_path)

        # Prepare data
        x_train, y_train = prepare_data(train_data)

        # Train the model
        model = train_model(x_train, y_train, n_estimators)

        # Save the trained model
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
   
if __name__ == "__main__":
    main()