import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from train_model import COLUMNS, download_dataset, train_and_save

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction")
st.caption("Machine-learning educational project | Python + Streamlit")

MODEL_PATH = Path("heart_model.joblib")
DATA_PATH = Path("data/heart.csv")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        if not DATA_PATH.exists():
            download_dataset(DATA_PATH)
        train_and_save(DATA_PATH, MODEL_PATH)
    return joblib.load(MODEL_PATH)

try:
    model_bundle = load_model()
    model = model_bundle["model"]
    metrics = model_bundle["metrics"]
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.info("Check your internet connection and run: python train_model.py")
    st.stop()

st.subheader("Enter patient measurements")
st.write("Use values provided by a qualified healthcare professional. Do not guess medical measurements.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 18, 100, 45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda x: {
            1:"Typical angina", 2:"Atypical angina", 3:"Non-anginal pain", 4:"Asymptomatic"}[x])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 70, 250, 120)
        chol = st.number_input("Serum cholesterol (mg/dl)", 80, 700, 200)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {
            0:"Normal", 1:"ST-T wave abnormality", 2:"Possible left ventricular hypertrophy"}[x])
        thalach = st.number_input("Maximum heart rate achieved", 50, 250, 150)
        exang = st.selectbox("Exercise-induced angina?", ["No", "Yes"])
    with c3:
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", [1, 2, 3])
        ca = st.selectbox("Major vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia test code", [3, 6, 7], format_func=lambda x: {
            3:"Normal (3)", 6:"Fixed defect (6)", 7:"Reversible defect (7)"}[x])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "age": age, "sex": 1 if sex == "Male" else 0, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg, "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }], columns=COLUMNS[:-1])
    prediction = int(model.predict(row)[0])
    st.subheader("Model output")
    if prediction == 1:
        st.warning("The model classified this input in the heart-disease-positive class.")
    else:
        st.success("The model classified this input in the heart-disease-negative class.")
    st.caption("This is a statistical model output, not a diagnosis or a measure of an individual's actual medical risk. Consult a qualified clinician.")

with st.expander("Model evaluation (held-out test set)"):
    st.write("The following metrics are calculated when the model is trained.")
    st.json(metrics)

st.divider()
st.caption("Educational demonstration only. The dataset is small and historical; performance on it does not establish clinical safety or suitability.")
