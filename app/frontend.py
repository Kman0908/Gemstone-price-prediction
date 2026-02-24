import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Gemstone Price Predictor")

st.title("💎 Gemstone Price Prediction")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("artifacts", "model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model
def load_preprocessor():
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    with open(preprocessor_path, "rb") as file:
        preprocessor = pickle.load(file)
    return preprocessor

model = load_model()
preprocessor = load_preprocessor()

# -----------------------------
# User Inputs
# -----------------------------

carat = st.number_input("Carat", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

depth = st.number_input("Depth", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

table = st.number_input("Table", min_value=0.0, max_value=100.0, value=55.0, step=0.1)

x = st.number_input("Length (x)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

y = st.number_input("Width (y)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

z = st.number_input("Height (z)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])

color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])

clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Price"):

    try:
        input_data = pd.DataFrame({
            "carat": [carat],
            "depth": [depth],
            "table": [table],
            "x": [x],
            "y": [y],
            "z": [z],
            "cut": [cut],
            "color": [color],
            "clarity": [clarity]
        })
        data_scaled = preprocessor.transform(input_data)
        prediction = model.predict(data_scaled)[0]

        st.success(f"💰 Estimated Price: ${round(prediction, 2)}")

    except Exception as e:
        st.error("Something went wrong during prediction.")
        st.write(e)