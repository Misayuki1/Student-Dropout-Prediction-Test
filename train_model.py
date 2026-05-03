import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def main():
    # 1. Load Dataset
    dataset_name = 'Tragico_student-dropout-dataset_v3.csv'
    print(f"Loading dataset: {dataset_name}...")
    
    if not os.path.exists(dataset_name):
        print(f"Error: {dataset_name} not found in current directory.")
        return

    try:
        df = pd.read_csv(dataset_name)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Preprocessing
    # Drop Student_ID if it exists
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        print("Dropped Student_ID column.")

    # Check for target column
    if 'Dropout' not in df.columns:
        print("Error: 'Dropout' target column not found.")
        return

    X = df.drop(columns=['Dropout'])
    y = df['Dropout']

    # Numerical and categorical features
    numerical_cols = [
        'Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate', 
        'Assignment_Delay_Days', 'Travel_Time_Minutes', 'Stress_Index', 
        'GPA', 'Semester_GPA', 'CGPA'
    ]
    
    categorical_cols = [
        'Gender', 'Internet_Access', 'Part_Time_Job', 'Scholarship', 
        'Semester', 'Department', 'Parental_Education'
    ]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # 3. Model Pipeline
    print("Building and training the model...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    pipeline.fit(X_train, y_train)

    # Accuracy
    print(f"Training Accuracy: {pipeline.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {pipeline.score(X_test, y_test):.4f}")

    # 4. Save
    joblib.dump(pipeline, 'pipeline.pkl')
    print("Success! Trained pipeline saved as 'pipeline.pkl'.")

if __name__ == '__main__':
    main()
