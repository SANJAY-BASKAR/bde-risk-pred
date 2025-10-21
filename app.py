from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap

# 1. Load pipeline objects
pipeline = joblib.load('big-data.pkl')
imputer = pipeline['imputer']
scaler = pipeline['scaler']
model = pipeline['model']
columns = pipeline['columns']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Ensure columns in correct order
    df = pd.DataFrame([data], columns=columns)
    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    prob = model.predict_proba(X_scaled)[:, 1][0]
    label = int(model.predict(X_scaled)[0])

    # Explain with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    feature_impact = dict(zip(columns, shap_values[0]))

    return jsonify({
        'probability': prob,
        'prediction': label,
        'feature_impact': feature_impact
    })

if __name__ == '__main__':
    app.run(debug=True)
