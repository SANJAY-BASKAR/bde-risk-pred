import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_pipeline(path): 
    return joblib.load(path)

# Load pipelines (ensure your .pkl files are in working dir)
stroke_pipe = load_pipeline('stroke_pipeline.pkl')
heart_pipe = load_pipeline('heart_disease_pipeline.pkl')
diabetes_pipe = load_pipeline('diabetes_pipeline.pkl')

columns = stroke_pipe['columns']  # assuming all pipelines share features

st.set_page_config(page_title="Health Risk Predictor", layout="wide")
st.title("Health Risk Prediction Dashboard with Explainable AI")

# Patient input section
st.sidebar.header("Patient Information")
input_data = {}

# Numeric continuous inputs with example defaults
input_data['age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=55)
input_data['HbA1c_level'] = st.sidebar.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=6.3)
input_data['blood_glucose_level'] = st.sidebar.number_input("Blood Glucose Level", min_value=40, max_value=400, value=105)
input_data['bmi'] = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
input_data['composite_risk_score'] = st.sidebar.number_input("Composite Risk Score", min_value=0.0, max_value=1.0, value=0.35)

# Categorical groups as radio buttons
residence = st.sidebar.radio("Residence Type", options=["Rural", "Urban"], index=1)
input_data['Residence_type_Rural'] = 1 if residence == "Rural" else 0
input_data['Residence_type_Urban'] = 1 if residence == "Urban" else 0

gender = st.sidebar.radio("Gender", options=["Male", "Female", "Other"], index=0)
input_data['gender_Male'] = 1 if gender == "Male" else 0
input_data['gender_Female'] = 1 if gender == "Female" else 0
input_data['gender_Other'] = 1 if gender == "Other" else 0

married = st.sidebar.radio("Ever Married?", options=["Yes", "No"], index=0)
input_data['ever_married_Yes'] = 1 if married == "Yes" else 0
input_data['ever_married_No'] = 1 if married == "No" else 0

hypertension = st.sidebar.radio("Hypertension?", options=["Yes", "No"], index=1)
input_data['hypertension'] = 1 if hypertension == "Yes" else 0

# Remove explicit Heart Disease input and default to No
input_data['heart_disease'] = 0

# Smoking status group
smoking = st.sidebar.selectbox(
    "Smoking Status",
    options=["never smoked", "formerly smoked", "smokes", "Unknown"],
    index=0
)
for status in ["never smoked", "formerly smoked", "smokes", "Unknown"]:
    input_data[f'smoking_status_{status}'] = 1 if smoking == status else 0

# Work type group
work_type = st.sidebar.selectbox(
    "Work Type",
    options=["Private", "Self-employed", "Govt_job", "Never_worked", "children"],
    index=0
)
for wt in ["Private", "Self-employed", "Govt_job", "Never_worked", "children"]:
    input_data[f'work_type_{wt}'] = 1 if work_type == wt else 0

# Age group
age_group = st.sidebar.selectbox("Age Group", options=["Young", "Middle", "Senior"], index=1)
input_data['age_group_young'] = 1 if age_group == "Young" else 0
input_data['age_group_middle'] = 1 if age_group == "Middle" else 0
input_data['age_group_senior'] = 1 if age_group == "Senior" else 0

# BMI group
bmi_group = st.sidebar.selectbox(
    "BMI Group",
    options=["Underweight", "Normal", "Overweight", "Obese"],
    index=2
)
input_data['bmi_underweight'] = 1 if bmi_group == "Underweight" else 0
input_data['bmi_normal'] = 1 if bmi_group == "Normal" else 0
input_data['bmi_overweight'] = 1 if bmi_group == "Overweight" else 0
input_data['bmi_obese'] = 1 if bmi_group == "Obese" else 0

# Fill zero for any missing feature columns
for col in columns:
    if col not in input_data:
        input_data[col] = 0

input_df = pd.DataFrame([input_data], columns=columns)

def predict_and_explain(pipe, input_df):
    imputer = pipe['imputer']
    scaler = pipe['scaler']
    model = pipe['model']

    X_imp = imputer.transform(input_df)
    X_scaled = scaler.transform(X_imp)
    prob = model.predict_proba(X_scaled)[:,1][0]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    return prob, shap_values, X_scaled

if st.sidebar.button("Predict"):
    st.subheader("Predicted Risks")
    probs = {}
    shap_values_dict = {}
    X_scaled_dict = {}

    with st.spinner("Computing predictions and explanations..."):
        probs['Stroke'], shap_values_dict['Stroke'], X_scaled_dict['Stroke'] = predict_and_explain(stroke_pipe, input_df)
        probs['Heart Disease'], shap_values_dict['Heart Disease'], X_scaled_dict['Heart Disease'] = predict_and_explain(heart_pipe, input_df)
        probs['Diabetes'], shap_values_dict['Diabetes'], X_scaled_dict['Diabetes'] = predict_and_explain(diabetes_pipe, input_df)

    for disease in ['Stroke', 'Heart Disease', 'Diabetes']:
        st.metric(label=f"{disease} Risk Probability", value=f"{probs[disease]:.2%}")

    st.subheader("Explainable AI Visualizations")

    def show_shap_summary(shap_values, X_scaled):
        fig = plt.figure(figsize=(8, 4))
        shap.summary_plot(shap_values, X_scaled, feature_names=columns, show=False)
        st.pyplot(fig)

    for disease in ['Stroke', 'Heart Disease', 'Diabetes']:
        st.markdown(f"### {disease} Feature Importance (Global)")
        show_shap_summary(shap_values_dict[disease], X_scaled_dict[disease])
        st.markdown("---")

    st.info("SHAP summary plots explain the influence of each feature on model predictions globally.")
