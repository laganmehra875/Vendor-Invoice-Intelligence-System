import joblib
from pathlib import Path
import os
import pandas as pd

from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_features
from modeling_evaluation import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_classifier
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)

    ## LOAD DATA
    print("Loading invoice data...")
    df = load_invoice_data()
    
    print("Applying labels...")
    df = apply_labels(df)

    ## prepare data 
    print("Splitting train and test sets...")
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    
    print("Scaling features...")
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, 'model/scaler.pkl'
    )

    ## Train and evaluate models 
    print("Tuning and training Random Forest Classifier via Grid Search...")
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        'Random Forest Classifier'
    )

    ## Save best model
    print("Saving the best model...")
    joblib.dump(grid_search.best_estimator_, 'model/predict_flag_invoice.pkl')
    print("Model saved to model/predict_flag_invoice.pkl successfully!")

if __name__ == "__main__":
    main()