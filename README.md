# Lipophilicity Model with Morgan Fingerprints

This repository contains a Python package and script to train a model predicting lipophilicity based on Morgan fingerprints of molecular SMILES representations. The model uses an MLP regressor and evaluates its performance using Root Mean Squared Error (RMSE) on test data.

## Contents
- **src/train_model.py**: Script for loading data, generating fingerprints, training the model, and saving evaluation results.
- **data/Lipophilicity.csv**: Data file.
- **config.json**: Model hyperparameters specification.
- **environment.yml**: Conda environment file listing dependencies.
- **results.txt**: Output file that stores the model's RMSE, the conda environment used, and key hyperparameters.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kbhalajiyadav/lipophilicity_model.git
   cd lipophilicity_model
   ```

2. **Set up the environment**:
   Create a new Conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate molecule_modeling  # Replace with your actual environment name
   ```

### Usage

You can specify model hyperparameters either through a JSON configuration file or by using command-line arguments.

#### Running the Script

To run the model training script with a JSON configuration file:
```bash
python src/train_model.py --config config.json
```

#### Specifying Hyperparameters Directly in the Command Line

If you prefer, you can specify hyperparameters directly on the command line. For example:
```bash
python src/train_model.py --hidden_layer_sizes 100,50 --alpha 0.01
```

#### Using Both JSON and Command-Line Arguments

When both are used, command-line arguments will override values from the JSON file:
```bash
python src/train_model.py --config config.json --alpha 0.01
```

#### Config File Example

Create a JSON configuration file like this in the main directory:
```json
{
   "hidden_layer_sizes": [100, 100],
   "alpha": 0.001
}
```

### Output

The script will save:
- The RMSE for the test set,
- The name of the active conda environment, and
- Hyperparameter settings

...to `results.txt` in the main directory.


- **Arguments**:
  - `--hidden_layer_sizes`: Specifies the architecture of the neural network.
  - `--alpha`: Sets the regularization strength of the model.

After execution, the script will output the test set RMSE, current environment, and hyperparameters to `results.txt`.

## License

This project is licensed under the Apache-2.0 License.

--
