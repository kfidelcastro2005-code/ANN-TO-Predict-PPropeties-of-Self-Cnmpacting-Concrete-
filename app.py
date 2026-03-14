import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load model and scaler
model = load_model("scc_ann_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("🧱 Self-Compacting Concrete Predictor (ANN)")
st.write("Enter mix proportions → Get instant compressive strength prediction")

# Input sliders (adjust ranges based on your dataset)
cement = st.slider("Cement (kg/m³)", 200, 600, 350)
fly_ash = st.slider("Fly Ash (kg/m³)", 0, 300, 100)
water = st.slider("Water (kg/m³)", 150, 250, 180)
sp = st.slider("Superplasticizer (kg/m³)", 0.0, 15.0, 5.0)
ca = st.slider("Coarse Aggregate (kg/m³)", 600, 1200, 850)
fa = st.slider("Fine Aggregate (kg/m³)", 600, 1000, 800)

if st.button("🔮 Predict Compressive Strength"):
    input_data = np.array([[cement, fly_ash, water, sp, ca, fa]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    
    st.success(f"Predicted 28-day Compressive Strength: **{prediction:.1f} MPa**")
    
    # Optional: show mix ratio summary
    st.write("Mix summary:")
    st.write(pd.DataFrame(input_data, columns=["Cement","Fly Ash","Water","SP","CA","FA"]))
