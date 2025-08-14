# ML-Project-E2E-2

This end-to-end Machine Learning project predicts student performance levels (High, Medium, Low) based on their academic scores and related features. It uses an XGBoost Classifier for robust multi-class classification, with full preprocessing, model training, and deployment.

# HOW TO SETUP
python -m venv .venv/

# HOW TO ACTIVATE
.venv\Scripts\activate

# RUN TRAINING SCRIPTS
python -m src.Pipeline.training_pipeline

# STREAMLIT APPLICATION
streamlit run streamlit_app/app.py

