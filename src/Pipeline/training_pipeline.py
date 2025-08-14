# training_pipeline.py

from src.models.train_model import main

processed_columns = [
    'gender', 'race/ethnicity', 'parental level of education',
    'lunch', 'test preparation course', 'math score', 
    'reading score', 'writing score'
]

if __name__ == "__main__":
    file_path = "data/processed/student_performance_encoded.csv"    
    target_column = "performance"
    main(file_path, target_column, processed_columns)
    
    