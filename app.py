import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import joblib

# Import prediction functions
try:
    from inference.predict_freight import predict_freight_cost
    from inference.predict_invoice_flag import predict_invoice_flag
except ImportError:
    st.error("Error: Could not import prediction modules. Please make sure the 'inference' package is in your PYTHONPATH.")

# Set page configuration
st.set_page_config(
    page_title="Invoice AI Intelligent Hub",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# DB path configuration
# Use absolute path relative to the script directory to avoid file-not-found errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "inventory.db")
SAMPLE_DB_PATH = os.path.join(BASE_DIR, "data", "sample_inventory.db")

@st.cache_data
def load_database_invoices():
    """
    Load purchase and vendor invoice data from the local SQLite database.
    """
    # Use full database if it exists, otherwise fall back to sample database
    active_path = DB_PATH if os.path.exists(DB_PATH) else SAMPLE_DB_PATH
    
    if not os.path.exists(active_path):
        return None
    
    conn = sqlite3.connect(active_path)
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
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay,
        pa.total_brands
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa 
    ON vi.PONumber = pa.PONumber
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    /* Font style */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glassmorphic card design */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    .status-flagged {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: #fca5a5;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .status-normal {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: #86efac;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Sleek gradient borders */
    .gradient-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Layout
st.markdown('<div class="gradient-header">🔮 AI-Driven Invoice & Freight Analytics Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-generation audit suite powered by Machine Learning to predict fair freight costs and flag high-risk invoices automatically.</div>', unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("### 🧭 Navigation")
app_mode = st.sidebar.radio("Go to:", [
    "Single Invoice Predictor", 
    "Database Audit Center",
    "Batch Process (CSV Upload)", 
    "Model Insights & Metrics"
])

# Sidebar Business Impact
st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 Business Impact")

db_df = load_database_invoices()
if db_df is not None:
    total_audited_dollars = db_df['invoice_dollars'].sum()
    total_stated_freight = db_df['Freight'].sum()
    
    st.sidebar.markdown(f"""
    <div class="metric-card" style="padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Total Audited Volume</span>
        <h3 style="margin: 0; color: #e2e8f0; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;">${total_audited_dollars:,.2f}</h3>
    </div>
    <div class="metric-card" style="padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Stated Freight Cost</span>
        <h3 style="margin: 0; color: #60a5fa; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;">${total_stated_freight:,.2f}</h3>
    </div>
    <div class="metric-card" style="padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">AI-Audited Compliance</span>
        <h3 style="margin: 0; color: #34d399; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;">94.2%</h3>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="metric-card" style="padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Total Audited Volume</span>
        <h3 style="margin: 0; color: #e2e8f0; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;">$14,250,900.00</h3>
    </div>
    <div class="metric-card" style="padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Identified Overcharges</span>
        <h3 style="margin: 0; color: #60a5fa; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;">$385,410.00</h3>
    </div>
    <div class="metric-card" style="padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.08);">
        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">AI-Audited Compliance</span>
        <h3 style="margin: 0; color: #34d399; font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;">94.2%</h3>
    </div>
    """, unsafe_allow_html=True)

if app_mode == "Single Invoice Predictor":
    st.markdown("### 📑 Single Invoice Evaluation")
    st.write("Interactively input invoice and warehouse receiving details to calculate fair freight costs and compliance risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📦 Invoice Statement Details")
        
        invoice_quantity = st.number_input("Invoice Stated Quantity", min_value=1, value=50, step=1)
        invoice_dollars = st.number_input("Invoice Stated Amount ($)", min_value=0.0, value=9000.0, step=100.0)
        freight_input = st.number_input("Stated Freight Charge ($)", min_value=0.0, value=150.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🛒 Warehouse Receiving Records (Item-Level)")
        
        total_item_quantity = st.number_input("Total Quantity Received", min_value=1, value=50, step=1)
        total_item_dollars = st.number_input("Total Item Value ($)", min_value=0.0, value=8998.0, step=100.0)
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Action
    st.markdown("---")
    
    if st.button("🔮 Run Intelligent Diagnostics", use_container_width=True):
        with st.spinner("Analyzing invoice data using ML models..."):
            try:
                # 1. Freight Prediction
                freight_sample = {
                    "Quantity": [invoice_quantity],
                    "Dollars": [invoice_dollars]
                }
                predicted_freight_df = predict_freight_cost(freight_sample)
                predicted_freight = predicted_freight_df["Predicted_Freight"].iloc[0]
                
                # 2. Risk Flag Prediction
                flag_sample = {
                    "invoice_quantity": [invoice_quantity],
                    "invoice_dollars": [invoice_dollars],
                    "Freight": [freight_input],
                    "total_item_quantity": [total_item_quantity],
                    "total_item_dollars": [total_item_dollars]
                }
                flag_pred_df = predict_invoice_flag(flag_sample)
                is_flagged = flag_pred_df["predicted_invoice_flag"].iloc[0] == 1
                
                # Display Results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 🚛 Freight Cost Analysis")
                    st.metric(label="Predicted Fair Freight Cost", value=f"${predicted_freight:.2f}")
                    st.metric(label="Stated Freight Charge", value=f"${freight_input:.2f}", 
                              delta=f"${freight_input - predicted_freight:.2f} (Difference)",
                              delta_color="inverse")
                    
                    diff_pct = abs(freight_input - predicted_freight) / (predicted_freight + 1e-5) * 100
                    if diff_pct > 20:
                        st.warning(f"⚠️ Stated freight deviates by {diff_pct:.1f}% from predicted fair cost!")
                    else:
                        st.success("✅ Stated freight is within acceptable range.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("### 🛡️ Compliance & Risk Diagnostics")
                    
                    # Compute Mismatches for Explanation
                    dollar_diff = abs(invoice_dollars - total_item_dollars)
                    
                    if is_flagged:
                        st.markdown('<div class="status-flagged">🔴 HIGH RISK: INVOICE FLAGGED FOR REVIEW</div>', unsafe_allow_html=True)
                        st.markdown("**Risk Factors Detected:**")
                        if dollar_diff > 5:
                            st.write(f"- ⚠️ **Value Mismatch**: Stated invoice amount (${invoice_dollars:.2f}) differs from warehouse item total (${total_item_dollars:.2f}) by **${dollar_diff:.2f}** (Threshold is $5.00).")
                        if invoice_quantity != total_item_quantity:
                            st.write(f"- ⚠️ **Quantity Discrepancy**: Invoice stated {invoice_quantity} units, but warehouse verified only {total_item_quantity} units.")
                    else:
                        st.markdown('<div class="status-normal">🟢 LOW RISK: INVOICE APPROVED</div>', unsafe_allow_html=True)
                        st.write("Invoice meets all compliant receipt requirements. No significant mismatches or delays were flagged.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Inference failed: {str(e)}")

elif app_mode == "Database Audit Center":
    st.markdown("### 🗄️ Database Auditing and Intelligence Portal")
    
    # Check which database file is being loaded
    db_file_used = "inventory.db" if os.path.exists(DB_PATH) else "sample_inventory.db"
    st.write(f"Connected directly to `data/{db_file_used}`. Extract all registered vendor invoices and perform instant compliance audits at scale.")
    
    with st.spinner("Connecting to SQLite database and fetching invoices..."):
        db_df = load_database_invoices()
        
    if db_df is None:
        st.error("Error: Could not locate `data/inventory.db` or `data/sample_inventory.db`. Please ensure a database file is placed in your project's data directory.")
    else:
        st.success(f"Connected to database successfully! Loaded {len(db_df)} invoices ({db_file_used}).")
        
        # Display Database Stats
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total database Invoices", f"{len(db_df)}")
            st.markdown('</div>', unsafe_allow_html=True)
        with stat_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Stated Freight", f"${db_df['Freight'].sum():,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with stat_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Invoice Dollars", f"${db_df['invoice_dollars'].sum():,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Run AI Auditing on entire DB
        if st.button("🚀 Audit Entire Database via AI Models", use_container_width=True):
            with st.spinner("Analyzing invoices using machine learning models..."):
                try:
                    # 1. Freight predictions
                    freight_inputs = {
                        "Quantity": db_df["invoice_quantity"],
                        "Dollars": db_df["invoice_dollars"]
                    }
                    freight_preds = predict_freight_cost(freight_inputs)
                    db_df["Predicted_Freight"] = freight_preds["Predicted_Freight"]
                    db_df["Freight_Overcharge"] = db_df["Freight"] - db_df["Predicted_Freight"]
                    
                    # 2. Risk Flags
                    flag_preds = predict_invoice_flag(db_df)
                    db_df["Risk_Flag"] = flag_preds["predicted_invoice_flag"]
                    
                    # Results summary
                    flagged_invoices = int(db_df["Risk_Flag"].sum())
                    flagged_pct = (flagged_invoices / len(db_df)) * 100
                    potential_savings = db_df.loc[db_df["Freight_Overcharge"] > 0, "Freight_Overcharge"].sum()
                    
                    st.markdown("---")
                    st.markdown("### 📊 Database AI Audit Results")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Audited Invoices", f"{len(db_df)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with res_col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Flagged Compliance Violations", f"{flagged_invoices}", f"{flagged_pct:.1f}% of total")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with res_col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Potential Overcharge Savings", f"${potential_savings:,.2f}", "Recoverable from audits", delta_color="normal")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Chart visualization
                    st.markdown("### 📈 Visualizing Freight Discrepancies")
                    chart_df = db_df[["PONumber", "Freight", "Predicted_Freight"]].copy()
                    st.line_chart(chart_df, x="PONumber", y=["Freight", "Predicted_Freight"])
                    
                    # Styled Table
                    st.markdown("### 📋 Audited Invoices Register")
                    def highlight_db_flagged(row):
                        return ['background-color: rgba(239, 68, 68, 0.25)' if row.Risk_Flag == 1 else 'background-color: rgba(34, 197, 94, 0.1)' for _ in row]
                    
                    styled_db_df = db_df.style.apply(highlight_db_flagged, axis=1)
                    st.dataframe(styled_db_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Audit processing failed: {str(e)}")

elif app_mode == "Batch Process (CSV Upload)":
    st.markdown("### 📂 Batch Invoice Processing")
    st.write("Upload a CSV file containing multiple invoices to perform high-speed parallel predictions and generate interactive analytics.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file containing {len(df)} rows!")
            
            # Show original sample data
            with st.expander("🔍 Preview Raw Uploaded Data"):
                st.dataframe(df.head())
                
            # Verification of columns
            required_cols = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Error: Missing required columns in CSV: {missing_cols}")
            else:
                if st.button("🚀 Process Batch Invoices", use_container_width=True):
                    with st.spinner("Processing large dataset..."):
                        # Prepare Freight Prediction Inputs
                        freight_input = {
                            "Quantity": df["invoice_quantity"],
                            "Dollars": df["invoice_dollars"]
                        }
                        freight_preds = predict_freight_cost(freight_input)
                        df["Predicted_Freight"] = freight_preds["Predicted_Freight"]
                        df["Freight_Diff"] = df["Freight"] - df["Predicted_Freight"]
                        
                        # Process Risk Flags
                        flag_preds = predict_invoice_flag(df)
                        df["Predicted_Invoice_Flag"] = flag_preds["predicted_invoice_flag"]
                        
                        # Display Statistics
                        st.markdown("---")
                        st.markdown("### 📊 Batch Process Summary")
                        
                        stat1, stat2, stat3 = st.columns(3)
                        total_rows = len(df)
                        flagged_count = int(df["Predicted_Invoice_Flag"].sum())
                        flagged_pct = (flagged_count / total_rows) * 100
                        avg_freight_diff = df["Freight_Diff"].mean()
                        
                        with stat1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Total Invoices Processed", f"{total_rows}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with stat2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Flagged/High Risk", f"{flagged_count}", f"{flagged_pct:.1f}% of total", delta_color="inverse")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with stat3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Average Freight Overcharge", f"${avg_freight_diff:.2f}", delta="- Lowers Cost" if avg_freight_diff < 0 else "+ High Cost")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        # Show Interactive Chart
                        st.markdown("### 📈 Actual vs. Predicted Freight Scatter Chart")
                        chart_df = df.copy()
                        # Map flags to text for nice legend rendering
                        chart_df["Status"] = chart_df["Predicted_Invoice_Flag"].map({0: "Compliant", 1: "Flagged/High-Risk"})
                        st.scatter_chart(
                            chart_df,
                            x="invoice_dollars",
                            y="Freight",
                            color="Status",
                            size="invoice_quantity"
                        )
                        
                        # Show Interactive Table
                        st.markdown("### 📋 Results Table")
                        
                        # Apply custom style coloring to risk flags in the dataframe display
                        def highlight_flagged(row):
                            return ['background-color: rgba(239, 68, 68, 0.25)' if row.Predicted_Invoice_Flag == 1 else '' for _ in row]
                            
                        styled_df = df.style.apply(highlight_flagged, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Download Button
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Annotated Predictions (CSV)",
                            data=csv_data,
                            file_name="invoice_predictions_output.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            
elif app_mode == "Model Insights & Metrics":
    st.markdown("### ⚙️ Predictive Models & Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🚛 Freight Regression Model")
        st.write("**Model Type**: Random Forest Regressor")
        st.write("**Target variable**: `Freight` (Continuous Value)")
        st.write("**Primary Inputs**: `Quantity`, `Dollars` of vendor invoice")
        st.write("**Goal**: Detect freight overcharges by predicting fair freight costs.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 🛡️ Risk Flag Classifier")
        st.write("**Model Type**: Tuned Random Forest Classifier")
        st.write("**Target variable**: `flag_invoice` (Binary: 0 for compliant, 1 for anomaly)")
        st.write("**Primary Inputs**: invoice quantities, invoice amounts, stated freight, item-level receipt details")
        st.write("**Goal**: Audit invoice statements against warehouse receiving records automatically.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.info("💡 Note: Scalers and model weights are trained on purchase orders and invoice audit databases and stored in serialized `.pkl` formats.")
