import streamlit as st
import joblib

# -------------------------
# Load pre-trained model and scaler
# -------------------------
model_ml = joblib.load("cricket_model.pkl")
scaler = joblib.load("cricket_scaler.pkl")

FEATURES = ["Ave", "HS", "century_rate", "fifty_rate", "duck_rate", "no_rate", "Mat"]

# -------------------------
# App title
# -------------------------
st.title("🏏 Cricket Player Performance Predictor")
st.markdown("Predict whether a cricket player is **Poor, Average, or Elite** based on stats.")

# -------------------------
# User input
# -------------------------
Ave = st.number_input("Batting Average (Ave)", value=30.0)
HS = st.number_input("Highest Score (HS)", value=90)
century_rate = st.number_input("Century Rate (per 100 innings)", value=0.0)
fifty_rate = st.number_input("Fifty Rate (per 100 innings)", value=5.0)
duck_rate = st.number_input("Duck Rate (per 100 innings)", value=3.0)
no_rate = st.number_input("Not-Out Percentage (%)", value=10.0)
Mat = st.number_input("Total Matches (Mat)", value=50)

# -------------------------
# Predict button
# -------------------------
if st.button("Predict Performance"):
    sample = [[Ave, HS, century_rate, fifty_rate, duck_rate, no_rate, Mat]]
    sample_scaled = scaler.transform(sample)

    pred_class = model_ml.predict(sample_scaled)[0]
    pred_proba = model_ml.predict_proba(sample_scaled)[0]

    st.subheader("Prediction Result")
    st.success(f"The predicted performance category is: **{pred_class}**")

    st.subheader("Probability per Class")
    for cls, prob in zip(model_ml.classes_, pred_proba):
        st.write(f"{cls}: {prob:.1%}")