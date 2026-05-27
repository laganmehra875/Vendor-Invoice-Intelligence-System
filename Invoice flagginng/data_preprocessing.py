import sqlite3
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_invoice_data(db_path: str = "/Users/mac/Documents/invoice Intelligent project/data/inventory.db") -> pd.DataFrame:
    """
    Load and join purchase and invoice data from the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    WITH purchase_agg AS (
        SELECT 
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        pa.avg_receiving_delay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa 
    ON vi.PONumber = pa.PONumber
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    """
    Apply risk labeling criteria:
    - High difference (> $5) between invoice and actual item total.
    - Abnormally high average receiving delay (> 10 days).
    """
    # Invoice total mismatch with item-level total 
    if abs(row['invoice_dollars'] - row['total_item_dollars']) > 5:
        return 1
    
    # Abnormally high receiving delay 
    if row["avg_receiving_delay"] > 10:
        return 1 
    
    return 0

def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the invoice risk labels to the DataFrame.
    """
    df['flag_invoice'] = df.apply(create_invoice_risk_label, axis=1)
    return df

def prepare_features(df: pd.DataFrame):
    """
    Label the data and split into features (X) and target (y).
    """
    # Create the risk label
    df = apply_labels(df)
    
    # Define features (X) and target (y)
    X = df[['invoice_quantity', 'invoice_dollars', 'Freight', 'total_item_quantity', 'total_item_dollars']]
    y = df['flag_invoice']
    
    return X, y

def split_data(df: pd.DataFrame, features: list, target: str, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train, X_test, scaler_save_path: str):
    """
    Scale features and save the scaler object.
    """
    # Ensure save directory exists
    dir_name = os.path.dirname(scaler_save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, scaler_save_path)
    return X_train_scaled, X_test_scaled

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets and apply standard scaling.
    Saves the scaler object for later inference usage.
    """
    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference/evaluation
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test