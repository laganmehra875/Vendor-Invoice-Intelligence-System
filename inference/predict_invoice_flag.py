import joblib 
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Possible paths for the model and scaler to ensure it runs from any working directory
MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, "Invoice flagginng", "model", "predict_flag_invoice.pkl"),
    "Invoice flagginng/model/predict_flag_invoice.pkl",
    "model/predict_flag_invoice.pkl",
    "models/predict_flag_invoice.pkl"
]

SCALER_PATHS = [
    os.path.join(PROJECT_ROOT, "Invoice flagginng", "model", "scaler.pkl"),
    "Invoice flagginng/model/scaler.pkl",
    "model/scaler.pkl",
    "models/scaler.pkl"
]

def load_file(paths):
    """
    Find and load a serialized joblib file from a list of potential paths.
    """
    for path in paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return joblib.load(f)
    raise FileNotFoundError(f"Could not locate file. Tried paths: {paths}")

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
        
    Parameter
    ---------
    input_data: dict or pd.DataFrame
    Returns
    -------
    pd.DataFrame with predicted invoice flag
    """
    # Load model and scaler dynamically
    model = load_file(MODEL_PATHS)
    scaler = load_file(SCALER_PATHS)
    
    input_df = pd.DataFrame(input_data)
    
    # Define features used during training
    features = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars"
    ]
    
    # Validate that all required features exist in input_df
    for feature in features:
        if feature not in input_df.columns:
            raise ValueError(f"Missing required feature: '{feature}'")
            
    # Ensure correct feature ordering
    X = input_df[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict flags
    input_df['predicted_invoice_flag'] = model.predict(X_scaled)
    return input_df

if __name__ == "__main__":
    ## Example inference run (local testing)
    sample_data = {
        "invoice_quantity": [100, 50, 30, 5],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [100, 50, 30, 5],
        "total_item_quantity": [100, 50, 30, 5],
        "total_item_dollars": [18500, 9000, 3000, 200]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)