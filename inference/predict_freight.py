import os
import joblib
import pandas as pd

# Use absolute path relative to this script's directory to ensure model can be loaded from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "predict_freight_model.pkl")


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
        
    Parameter
    ---------
    input_data: dict or pd.DataFrame
    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    
    # Ensure the features match the training features in order and presence
    features = ['Quantity', 'Dollars']
    for feature in features:
        if feature not in input_df.columns:
            raise ValueError(f"Missing required feature: '{feature}'")
            
    # Reorder columns to match the training features
    input_df = input_df[features]
    
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    ## Example inference run (local testing)
    sample_data = {
        "Quantity": [100, 50, 30, 5],
        "Dollars": [18500, 9000, 3000, 200]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)