import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# ---------------------------
# Load trained model & scaler
# ---------------------------
model = tf.keras.models.load_model("ann_model_all_features.keras")
scaler = pickle.load(open("scaler_all.pkl", "rb"))

# ---------------------------
# App title
# ---------------------------
st.title("Breast Cancer Diagnosis Prediction (ANN)")
st.write("Enter feature values below to predict whether the tumor is Malignant (M) or Benign (B).")

# ---------------------------
# Feature list (all 31 features)
# ---------------------------
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst' 
]

# ---------------------------
# Input fields for user
# ---------------------------
st.header("Input Feature Values")
user_input = []
for feature in features:
    val = st.number_input(f"{feature}", min_value=0.0, value=0.0, format="%.5f")
    user_input.append(val)

# Convert to array and scale
input_data = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_data)

# ---------------------------
# Prediction button
# ---------------------------
if st.button("Predict Diagnosis"):
    pred_prob = model.predict(input_scaled)[0][0]
    pred_class = "Malignant (M)" if pred_prob > 0.5 else "Benign (B)"
    
    st.subheader(f"Predicted Diagnosis: {pred_class}")
    st.write(f"Prediction Probability: {pred_prob:.4f}")
