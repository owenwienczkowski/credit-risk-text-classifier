# Credit Risk Classification Pipeline

A modular machine learning pipeline designed to predict loan default risk using structured tabular data. The system reflects industry-standard approaches used in credit scoring and financial risk assessment.

## Project Overview
- **Objective:** Binary classification of loan default risk
- **Dataset:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Target Variable:** `loan_status`
- **Features:** Borrower demographics, loan characteristics, credit history

## Pipeline Structure
Modularized for clarity and extensibility:
- `load_data.py`: Downloads and loads the dataset
- `preprocess.py`: Handles missing values, encodes features, removes outliers
- `train.py`: Trains models using Scikit-learn Pipelines with scaling
- `evaluate.py`: Generates performance metrics and visualizations
- `inference.py`: Loads saved models for prediction
- `scripts/run_pipeline.py`: Executes the full end-to-end workflow

## Models Implemented
- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation
- Stratified train/test split to preserve class balance
- Metrics reported: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Visual outputs: Confusion matrix, ROC curve, Precision-Recall curve

## Outputs
- Trained models saved to `/models/`
- Evaluation plots saved to `/outputs/`

## Key Highlights
- Fully modular and reproducible implementation using Scikit-learn Pipelines
- Incorporates best practices for preprocessing, tuning, and model evaluation
- Suited for credit risk prediction and similar tabular classification tasks

## How to Run
```bash
# Run the complete training and evaluation pipeline
python -m scripts.run_pipeline