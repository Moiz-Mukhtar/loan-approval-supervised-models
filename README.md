# Supervised Machine Learning Models for Loan Approval Prediction

## Dataset Name & Source
- **Dataset:** Loan Prediction Problem Dataset (`loan_data.csv`)
- **Source:** [Kaggle - Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

## Project Summary
This project implements and evaluates core supervised machine learning models—K-Nearest Neighbors (KNN), Decision Trees, Logistic Regression, and Linear Regression—on residential loan application data (614 rows, 13 columns). The pipeline includes leak-free data preprocessing (`SimpleImputer`, `StandardScaler`, and `OneHotEncoder` via `ColumnTransformer`), hyperparameter tuning for KNN ($K = 1$ to $15$) and Decision Tree depth (`max_depth`), and evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrices.

## Best Model & Performance
- **Best Classification Model:** Logistic Regression (tied with tuned Decision Tree at `max_depth=2`)
- **Test Accuracy:** `86.18%` (`0.8618`)
- **Test F1-Score:** `0.9081`

## How to Run the Notebook
1. Clone this repository:
   ```bash
   git clone [https://github.com/Moiz-Mukhtar/loan-approval-supervised-models.git](https://github.com/Moiz-Mukhtar/loan-approval-supervised-models.git)
   cd loan-approval-supervised-models

## Install the required dependencies:
    pip install -r requirements.txt

Launch Jupyter Notebook or open supervised_model_evaluation.ipynb in VS Code / Google Colab and run all cells top to bottom.