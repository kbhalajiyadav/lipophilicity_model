# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import os

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else np.zeros(n_bits)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_layer_sizes", type=int, nargs='+', default=[100, 100], help="Hidden layers configuration")
parser.add_argument("--alpha", type=float, default=0.0001, help="Regularization term for MLPRegressor")
args = parser.parse_args()

# Load and preprocess data
data = pd.read_csv('Lipophilicity.csv')  # Ensure the dataset is in the same directory
X = data['smiles']
y = data['exp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate fingerprints
X_train_fps = np.array([np.array(generate_morgan_fingerprint(sm)) for sm in X_train])
X_test_fps = np.array([np.array(generate_morgan_fingerprint(sm)) for sm in X_test])

# Scale target
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# Model training
mlp = MLPRegressor(hidden_layer_sizes=tuple(args.hidden_layer_sizes), alpha=args.alpha, max_iter=1000, random_state=42)
mlp.fit(X_train_fps, y_train_scaled)

# Make predictions and calculate RMSE
y_pred_scaled = mlp.predict(X_test_fps)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Get environment name
conda_env = os.getenv("CONDA_DEFAULT_ENV", "base")

# Save results
with open("results.txt", "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"Conda Environment: {conda_env}\n")
    f.write(f"Hyperparameters: hidden_layer_sizes={args.hidden_layer_sizes}, alpha={args.alpha}\n")

print(f"RMSE for Morgan fingerprints: {rmse}")
print(f"Results saved to results.txt")