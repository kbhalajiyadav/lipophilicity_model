# src/train_model.py

import pandas as pd
import numpy as np
import argparse
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem

# Argument parsing for hyperparameters
parser = argparse.ArgumentParser(description="Train an MLP model with Morgan fingerprints on the Lipophilicity dataset.")
parser.add_argument("--config", type=str, help="Path to JSON config file with hyperparameters.")
parser.add_argument("--hidden_layer_sizes", type=str, help="Comma-separated layer sizes for the neural network, e.g., 100,100.")
parser.add_argument("--alpha", type=float, help="Regularization strength for the model.")
args = parser.parse_args()

# Load hyperparameters from JSON config file if provided
if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)
    hidden_layer_sizes = tuple(config.get("hidden_layer_sizes", [100, 100]))  # Default to [100, 100]
    alpha = config.get("alpha", 0.001)  # Default to 0.001
else:
    # Default values if no config is provided and no command-line arguments are given
    hidden_layer_sizes = (100, 100)
    alpha = 0.001

# Override JSON settings with command-line arguments if provided
if args.hidden_layer_sizes:
    hidden_layer_sizes = tuple(int(x) for x in args.hidden_layer_sizes.split(","))
if args.alpha is not None:
    alpha = args.alpha

# Load the dataset
data = pd.read_csv('Lipophilicity.csv')
X = data['smiles']
y = data['exp']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to generate Morgan fingerprints
def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return np.zeros(n_bits)

# Apply Morgan fingerprints generation on train and test sets
X_train_morgan = np.array([np.array(generate_morgan_fingerprint(smile)) for smile in X_train])
X_test_morgan = np.array([np.array(generate_morgan_fingerprint(smile)) for smile in X_test])

# Scale the target variable
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# Initialize the MLPRegressor with parsed hyperparameters
mlp_morgan = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=1000, random_state=42)

# Train the model
mlp_morgan.fit(X_train_morgan, y_train_scaled.ravel())

# Make predictions on the test set and inverse transform the predictions
y_pred_morgan_scaled = mlp_morgan.predict(X_test_morgan)
y_pred_morgan = scaler.inverse_transform(y_pred_morgan_scaled.reshape(-1, 1))

# Calculate RMSE
rmse_morgan = np.sqrt(mean_squared_error(y_test, y_pred_morgan))

# Get the Conda environment name
conda_env_name = os.getenv("CONDA_DEFAULT_ENV")

# Save results to a file
with open("results.txt", "w") as f:
    f.write(f"RMSE for Morgan fingerprints: {rmse_morgan}\n")
    f.write(f"Conda Environment: {conda_env_name}\n")
    f.write(f"Hyperparameters: hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}\n")

print(f'RMSE for Morgan fingerprints: {rmse_morgan}')
print(f'Conda Environment: {conda_env_name}')
print(f'Hyperparameters: hidden_layer_sizes={hidden_layer_sizes}, alpha={alpha}')