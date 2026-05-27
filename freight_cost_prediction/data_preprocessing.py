import sqlite3
from sklearn.model_selection import train_test_split 
import pandas as pd 

def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """
    Load vendor invoice data from a SQLite database.
    """
    conn=sqlite3.connect(db_path)
    query="SELECT * FROM vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def prepare_feature(df: pd.DataFrame):
    """
    Select feature and target variables.
    """
    X = df[['Quantity', 'Dollars']]
    y = df['Freight']
    return X, y

def split_data(X,y,test_size=0.2,random_state=42):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(X,y,test_size=test_size,random_state=random_state)
    

    

   