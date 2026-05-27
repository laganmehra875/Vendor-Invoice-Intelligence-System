# 🔮 Invoice AI Intelligent Hub

Next-generation AI-driven audit suite powered by Machine Learning to predict fair freight costs and flag high-risk invoice anomalies automatically.

This application provides procurement, finance, and logistics departments with automated tools to evaluate vendor invoice compliance, identify freight overcharges, and audit transaction records against actual warehouse receiving logs.

---
# 🚀 Streamlit Application 

```bash
https://vendor-invoice-intelligence-system-mguunoecrprbi86zjyjutp.streamlit.app/
```


https://github.com/laganmehra875/Vendor-Invoice-Intelligence-System/blob/main/Vendor-Invoice-Intelligence-System%20Warehouse.png

---

## 🚀 Key Features

* **🚛 Freight Cost Regressor**: Evaluates and predicts fair market freight expenses based on invoice quantity and total monetary volume.
* **🛡️ Risk Flag Classifier**: Evaluates compliance risk (e.g. price/quantity mismatches, long delays) between vendor invoices and verified warehouse purchase logs.
* **🗄️ Database Audit Center**: Directly queries the local SQLite inventory database (`data/inventory.db`), performs automated batch ML audits, visualizes cost trends, and flags discrepancies.
* **📂 CSV Batch Processing**: Drag-and-drop file uploader to analyze external invoice registers, render dynamic multi-dimensional scatter charts, and export annotated results.
* **💼 Business Impact Dashboard**: Side-panel navigation display tracking total audited transaction volumes, historical freight expense, and automated compliance ratios in real-time.
* **🎨 Premium Dark Theme UI**: A high-tech interactive interface designed with futuristic glassmorphism aesthetics, gradient elements, and micro-animations.

---

## 📁 Project Structure

```text
invoice Intelligent project/
│
├── app.py                            # Streamlit Web Application (AI Dashboard)
├── README.md                         # Project documentation
│
├── data/
│   └── inventory.db                  # Local SQLite database containing PO & Invoice records
│
├── models/
│   └── predict_freight_model.pkl     # Best trained regression model for freight costing
│
├── Invoice flagginng/                # Classifier training pipeline
│   ├── data_preprocessing.py         # DB queries, feature scaling and split orchestrations
│   ├── modeling_evaluation.py       # Classifier model definitions and hyperparameter grid-search
│   ├── train.py                      # Main pipeline execution to train & save classification model
│   └── model/
│       ├── predict_flag_invoice.pkl  # Trained Random Forest classifier
│       └── scaler.pkl                # Standard Scaler for feature normalization
│
├── freight_cost_prediction/          # Regressor training pipeline
│   ├── data_preprocessing.py         # DB loading and feature selection for regression
│   ├── train.py                      # Pipeline executor training Linear/Tree/Forest regressors
│   └── model_evaluation.py           # Regressor evaluation metrics
│
└── inference/                        # Inference serving scripts
    ├── predict_freight.py            # Serves freight predictions with automatic feature matching
    └── predict_invoice_flag.py       # Serves risk flags with dynamic path scaling & loading
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed. 

### 2. Environment Setup
Install the required machine learning, database, and front-end dependencies:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 3. Database Verification
Ensure the SQLite database file `inventory.db` is correctly located inside the `data/` directory at the project root.

---

## 🏋️‍♂️ Model Training Pipelines

If you wish to re-train the machine learning models on updated dataset:

### 1. Train the Freight Cost Regressor
This pipeline loads vendor invoices, trains Linear Regression, Decision Tree, and Random Forest models, and automatically exports the model with the lowest Mean Absolute Error (MAE):
```bash
python "freight_cost_prediction/train.py"
```
*Model will be saved to:* `models/predict_freight_model.pkl`

### 2. Train the Invoice Compliance Classifier
This pipeline aggregates transaction data, applies business anomaly labels (e.g. price deviation > $5), fits a tuned Random Forest Classifier, and saves both the model and the fitted StandardScaler:
```bash
python "Invoice flagginng/train.py"
```
*Models will be saved to:* `Invoice flagginng/model/predict_flag_invoice.pkl` and `Invoice flagginng/model/scaler.pkl`

---

## 🔮 Running Inference

You can run offline test predictions using local Python scripts:

### Test Freight Cost Predictor:
```bash
python inference/predict_freight.py
```

### Test Invoice Risk Flagging:
```bash
python inference/predict_invoice_flag.py
```

---

## 🖥️ Launching the Web Dashboard

To run the interactive Streamlit application, execute:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the system.

---

## 🧠 Behind the Models

### 🚛 Freight Regressor
* **Algorithm**: Random Forest Regressor
* **Primary Features**: `Quantity` (invoiced), `Dollars` (invoice amount)
* **Goal**: Predicts expected freight expenses to evaluate vendor carriage markups.

### 🛡️ Compliance Classifier
* **Algorithm**: Tuned Random Forest Classifier (GridSearchCV)
* **Primary Features**: `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars`
* **Goal**: Verifies statements against physical warehouse receiving records to flag discrepancies.
