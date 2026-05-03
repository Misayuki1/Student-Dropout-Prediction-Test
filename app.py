from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained pipeline when the app starts
# This avoids reloading the model on every request, improving performance
pipeline_path = 'pipeline.pkl'

if os.path.exists(pipeline_path):
    pipeline = joblib.load(pipeline_path)
    print(f"Loaded pipeline from {pipeline_path}")
else:
    print(f"Warning: {pipeline_path} not found. Please run train_model.py first.")
    pipeline = None

@app.route('/')
def home():
    """Render the home page with the predictive form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission, make a prediction, and render the result."""
    if not pipeline:
        return render_template('index.html', error_text="Model pipeline not loaded. Please train the model first.")

    if request.method == 'POST':
        try:
            # Safely extract all 17 features from the HTML form
            # Wrap inputs in lists to create a single-row DataFrame later
            input_data = {
                # Numerical Features
                'Age': [float(request.form.get('Age', 0))],
                'Family_Income': [float(request.form.get('Family_Income', 0))],
                'Study_Hours_per_Day': [float(request.form.get('Study_Hours_per_Day', 0))],
                'Attendance_Rate': [float(request.form.get('Attendance_Rate', 0))],
                'Assignment_Delay_Days': [float(request.form.get('Assignment_Delay_Days', 0))],
                'Travel_Time_Minutes': [float(request.form.get('Travel_Time_Minutes', 0))],
                'Stress_Index': [float(request.form.get('Stress_Index', 0))],
                'GPA': [float(request.form.get('GPA', 0))],
                'Semester_GPA': [float(request.form.get('Semester_GPA', 0))],
                'CGPA': [float(request.form.get('CGPA', 0))],
                
                # Categorical Features
                'Gender': [request.form.get('Gender', '')],
                'Internet_Access': [request.form.get('Internet_Access', '')],
                'Part_Time_Job': [request.form.get('Part_Time_Job', '')],
                'Scholarship': [request.form.get('Scholarship', '')],
                'Semester': [str(request.form.get('Semester', ''))], # Keep categorical
                'Department': [request.form.get('Department', '')],
                'Parental_Education': [request.form.get('Parental_Education', '')]
            }
            
            # Convert dictionary into a Pandas DataFrame
            # Column names must exactly match the ones used during training
            df_input = pd.DataFrame(input_data)
            
            # Pass the DataFrame through the loaded pipeline to get prediction
            prediction = pipeline.predict(df_input)[0]
            
            # Interpret the prediction result
            if prediction == 1:
                result = "High Risk of Dropout"
            else:
                result = "Low Risk of Dropout"
                
            # Render the same page, but this time pass the prediction text to display it
            return render_template('index.html', prediction_text=result)
            
        except ValueError as ve:
             return render_template('index.html', error_text=f"Value Error: Please ensure all numeric fields contain valid numbers.")
        except Exception as e:
            # Catch unexpected errors to prevent application crash
            return render_template('index.html', error_text=f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices (like your tablet) on the same Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True)
