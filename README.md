# Household Appliance Energy Prediction using MLP

An end-to-end regression project that estimates household appliance energy use (`Appliances`, Wh) from the official UCI Appliances Energy Prediction dataset.

## Run on Windows

From the project directory in the VS Code PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
streamlit run app.py
```

`python main.py` can also receive `--learning-rate`, `--epochs`, and `--patience` options. If dataset download is unavailable, save the official `energydata_complete.csv` file to `data\energydata_complete.csv`. The dataset loader validates the target and timestamp columns and does not substitute synthetic data.

## Pipeline and artifacts

- Validates and chronologically sorts the UCI CSV, checks data types, missing values, and duplicates.
- Creates a chronological 70% training / 15% validation / 15% test split.
- Adds cyclical hour, weekday, and month features; excludes the dataset's random `rv1`/`rv2` fields; fits median imputation and feature/target scaling on training rows only.
- Trains a 128 → 64 → 32 ReLU MLP using explicit full-batch gradient descent: the complete training matrix is used for each gradient, followed by exactly one parameter update each epoch. Validation early stopping restores the best checkpoint.
- Compares the MLP with a baseline that predicts the training-target mean. Reports MAE, RMSE, and R² in Wh on identical held-out test observations.
- Saves the model and preprocessing bundle in `models\`; metrics, epoch history, inspection data, predictions, and report in `outputs\`; and EDA/evaluation figures in `outputs\figures\`.

The Streamlit dashboard has project overview, dataset explorer, model training, model evaluation, energy prediction, and reports/downloads sections. Prediction inputs follow the model's saved feature names and preprocessing order.

Dataset citation: Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VC8G. UCI lists the dataset under CC BY 4.0; retain attribution when redistributing dataset material.
