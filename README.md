# Customer Churn Prediction

This project trains and compares machine-learning models to predict customer churn using the Telco Customer Churn dataset. It also ranks churn-related features and assigns churn-risk tiers to customers in the test set.

## Requirements

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost

## Setup

Run these commands from the project directory:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib scikit-learn xgboost
```

## Run

Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project directory, then run:

```bash
python churn.py
```

The script loads and cleans the data, encodes categorical features, and splits it into training and test sets. It trains Logistic Regression, Random Forest, and XGBoost classifiers, then prints ROC-AUC scores, confusion matrices, and classification reports for each model.

It also prints a normalized feature-importance ranking and a churn-risk summary for the test set, and displays a chart of the top churn drivers. The risk tiers are Low (probability up to 0.4), Medium (over 0.4 to 0.7), and High (over 0.7).
