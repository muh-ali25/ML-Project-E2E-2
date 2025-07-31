# src/models/train_model.py

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Load Data ---
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# --- Create Target Column ---
def create_target_column(df):
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['performance'] = df['average_score'].apply(lambda x: 'High' if x >= 80 else ('Medium' if x >= 50 else 'Low'))
    df.drop(columns=['average_score'], inplace=True)
    return df

# --- Encode Categorical Features ---
def encode_features(df):
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include='object').columns:
        if col != 'performance':  # Don't encode the target yet
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    return df_encoded, label_encoders

# --- Encode Target Separately ---
def encode_target(df, target_col='performance'):
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])
    return df, target_encoder

# --- Split Data ---
def split_data(df, target_col='performance'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model

# --- Evaluate Model ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    
# ---- Save Model -----

def save_model(model_directory: str, model_name:str, model:object):

    try:
        os.makedirs(model_directory, exist_ok=True)
        model_path = os.path.join(model_directory, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        print("Error in Saving Model File\t", e)
        raise e   
    
# ----- Load Model -----

def load_model(model_directory: str, model_name:str):

    try:
        model_path = os.path.join(model_directory, f"{model_name}.pkl") 
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except Exception as e:
        print("Error in Loading Model File\t", e)
        return None     

# --- Main method to be called from training_pipeline.py ---
def main(file_path, target_col, processed_columns=None):
    df = load_data(file_path)
    df = create_target_column(df)
    df_encoded, _ = encode_features(df)
    df_encoded, _ = encode_target(df_encoded, target_col)
    X_train, X_test, y_train, y_test = split_data(df_encoded, target_col)
    model = train_xgboost(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model_directory= "artifacts", model_name="xgboost classifier", model= model)
