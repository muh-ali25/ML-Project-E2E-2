import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.train_model import load_model


# Load trained model from artifacts
model = load_model("artifacts", "xgboost classifier")

# Map encoded prediction back to label
label_map = {0: 'High', 1: 'Low', 2: 'Medium'}

st.title("🎓 Student Performance Predictor")

# Input fields
gender = st.selectbox("Gender", ['female', 'male'])
race_ethnicity = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ['standard', 'free/reduced'])
test_prep = st.selectbox("Test Preparation Course", ['none', 'completed'])
math_score = st.number_input("Math Score", 0, 100)
reading_score = st.number_input("Reading Score", 0, 100)
writing_score = st.number_input("Writing Score", 0, 100)

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Please check the path or file.")
    else:
        # Prepare input data
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

        # Apply same average score logic
        input_df['average_score'] = input_df[['math score', 'reading score', 'writing score']].mean(axis=1)
        input_df['performance'] = input_df['average_score'].apply(
            lambda x: 'High' if x >= 80 else ('Medium' if x >= 50 else 'Low')
        )
        input_df.drop(columns=['average_score'], inplace=True)

        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        input_encoded = input_df.copy()
        for col in input_encoded.select_dtypes(include='object').columns:
            if col != 'performance':
                le = LabelEncoder()
                input_encoded[col] = le.fit_transform(input_encoded[col])

        X_input = input_encoded.drop(columns=['performance'], errors='ignore')

        # Make prediction
        pred = model.predict(X_input)[0]
        performance_label = label_map[pred]

        st.success(f"Predicted Student Performance: **{performance_label}**")
