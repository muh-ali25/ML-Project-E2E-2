import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sys
from sklearn.preprocessing import LabelEncoder

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.train_model import load_model

# Load model
model = load_model("artifacts", "xgboost classifier")
label_map = {0: 'High', 1: 'Low', 2: 'Medium'}

# --- Page Config ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #4A6EE0;
            text-align: center;
        }
        .css-1d391kg {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1>Student Performance Predictor</h1>", unsafe_allow_html=True)
st.write("Fill in the student details below to predict the expected performance level (Low, Medium, or High).")

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['female', 'male'])
        parent_edu = st.selectbox("Parental Level of Education", [
            "some high school", "high school", "some college", "associate's degree",
            "bachelor's degree", "master's degree"
        ])
        test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
        math_score = st.number_input("Math Score", 0, 100, step=1)
    with col2:
        race_ethnicity = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
        lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
        reading_score = st.number_input("Reading Score", 0, 100, step=1)
        writing_score = st.number_input("Writing Score", 0, 100, step=1)

    submitted = st.form_submit_button("Predict")

# --- Prediction ---
if submitted:
    if model is None:
        st.error("Model not loaded. Please check the path or file.")
    else:
        # Prepare data
        input_df = pd.DataFrame([{
            'gender': gender,
            'race/ethnicity': race_ethnicity,
            'parental level of education': parent_edu,
            'lunch': lunch,
            'test preparation course': test_prep,
            'math score': math_score,
            'reading score': reading_score,
            'writing score': writing_score
        }])

        input_df['average_score'] = input_df[['math score', 'reading score', 'writing score']].mean(axis=1)
        input_df['performance'] = input_df['average_score'].apply(
            lambda x: 'High' if x >= 80 else ('Medium' if x >= 50 else 'Low')
        )
        input_df.drop(columns=['average_score'], inplace=True)

        # Encode categorical features
        input_encoded = input_df.copy()
        for col in input_encoded.select_dtypes(include='object').columns:
            if col != 'performance':
                le = LabelEncoder()
                input_encoded[col] = le.fit_transform(input_encoded[col])

        X_input = input_encoded.drop(columns=['performance'], errors='ignore')

        # Predict
                
        pred = model.predict(X_input)[0]
        performance_label = label_map[pred]

        st.markdown(f"""
            <div style="
                background-color: #e6f0ff;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                text-align: center;
                border: 1px solid #cce0ff;
            ">
                <h2 style="color: #003366; font-weight: bold; margin: 0;">
                    Predicted Student Performance: {performance_label}
                </h2>
            </div>
        """, unsafe_allow_html=True)

        


