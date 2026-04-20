import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# -------------------------------
# Scaling function (IMPORTANT)
# Converts 0–100 → -0.1 to 0.1
# -------------------------------
def scale_input(value):
    return (value / 100) * 0.2 - 0.1


# -------------------------------
# UI
# -------------------------------
st.title("🩺 Diabetes Progression Prediction App")

st.write("Enter patient details in sidebar and click Predict")

st.sidebar.header("Patient Data")

st.sidebar.info("0 = Low | 50 = Average | 100 = High")

# -------------------------------
# Inputs (0–100 UI)
# -------------------------------
age = scale_input(st.sidebar.slider("Age", 0, 100, 50))

# Sex (categorical - better handled separately)
sex_option = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = -0.05 if sex_option == "Female" else 0.05

bmi = scale_input(st.sidebar.slider("Body Mass Index (BMI)", 0, 100, 50))
bp = scale_input(st.sidebar.slider("Blood Pressure", 0, 100, 50))

s1 = scale_input(st.sidebar.slider("Total Cholesterol", 0, 100, 50))
s2 = scale_input(st.sidebar.slider("LDL (Bad Cholesterol)", 0, 100, 50))
s3 = scale_input(st.sidebar.slider("HDL (Good Cholesterol)", 0, 100, 50))
s4 = scale_input(st.sidebar.slider("Cholesterol Ratio", 0, 100, 50))
s5 = scale_input(st.sidebar.slider("Triglycerides", 0, 100, 50))
s6 = scale_input(st.sidebar.slider("Blood Sugar Level", 0, 100, 50))


# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")
    st.write(f"Predicted Score: {prediction:.2f}")

    # Risk levels
    if prediction < 100:
        st.success("🟢 Low Risk")
    elif prediction < 200:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    st.info("Higher score means higher diabetes progression risk.")