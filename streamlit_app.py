import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('big-data.pkl')

pipeline = load_model()
imputer = pipeline['imputer']
scaler = pipeline['scaler']
model = pipeline['model']
columns = pipeline['columns']

# Theme customization (create or modify ~/.streamlit/config.toml for persistent theme)
# Or inline styling with markdown options

# Page title
st.title("🏥 Stroke Risk Prediction System")
st.write("Enter patient info or upload batch data for AI-driven stroke risk assessment.")

# Patient data input (dynamic for all features)
st.sidebar.header("Patient Data Entry")
input_data = {}
for col in columns:
    if col.startswith('age') or col.startswith('HbA1c') or col.startswith('blood_glucose') or col.startswith('composite'):
        # Numeric input
        input_data[col] = st.sidebar.number_input(
            label=col.replace('_', ' ').title(),
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )
    elif 'married' in col or 'hypertension' in col or 'heart_disease' in col or 'smokes' in col or 'other' in col:
        # Binary
        input_data[col] = st.sidebar.selectbox(
            label=col.replace('_', ' ').title(),
            options=[0, 1]
        )
    elif 'gender' in col or 'work_type' in col or 'residence_type' in col or 'smoking_status' in col:
        # Categorical (one-hot)
        input_data[col] = st.sidebar.selectbox(
            label=col.replace('_', ' ').title(),
            options=[0, 1]
        )

# Fill missing features with 0
for col in columns:
    if col not in input_data:
        input_data[col] = 0

# Predict Button
if st.sidebar.button("🔍 Predict risk for entered data"):
    df_input = pd.DataFrame([input_data], columns=columns)
    X_imp = imputer.transform(df_input)
    X_scaled = scaler.transform(X_imp)

    # Prediction
    prob = float(model.predict_proba(X_scaled)[:,1][0])
    pred = int(model.predict(X_scaled)[0])

    # Show results
    st.subheader("🚨 Risk Prediction")
    st.metric("Risk Probability", f"{prob:.2%}")
    risk_level = "High" if pred==1 else "Low"
    st.metric("Risk Level", risk_level)
    st.progress(prob)

    # Interpretations
    if prob > 0.7:
        st.warning("⚠️ High risk! Immediate consultation advised.")
    elif prob > 0.3:
        st.info("⚡ Moderate risk. Monitor closely.")
    else:
        st.success("✅ Low risk. Keep healthy!")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    st.subheader("Feature Contribution (SHAP)")
    shap.summary_plot(shap_values, pd.DataFrame(X_scaled, columns=columns), show=False)
    st.pyplot(bbox_inches='tight')

# Patient history tracking
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.sidebar.button("Add current prediction to history"):
    record = {
        **input_data,
        "probability": prob,
        "risk_level": risk_level
    }
    st.session_state["history"].append(record)

if st.session_state["history"]:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state["history"]))

# Batch prediction with file upload
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV for batch predictions", type="csv")

if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)
    df_batch = df_batch.reindex(columns=columns, fill_value=0)
    X_batch = imputer.transform(df_batch)
    X_batch_scaled = scaler.transform(X_batch)
    probs = model.predict_proba(X_batch_scaled)[:,1]
    preds = model.predict(X_batch_scaled)
    results = df_batch.copy()
    results["Probability"] = probs
    results["Prediction"] = preds

    st.subheader("Batch Prediction Results")
    st.dataframe(results)

    # Download link
    csv = results.to_csv(index=False).encode()
    st.download_button("Download Results as CSV", data=csv, file_name="batch_results.csv", mime="text/csv")
