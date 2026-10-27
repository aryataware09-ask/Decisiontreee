import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Decision Tree model
MODEL_PATH = "Decision.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Embedded HTML Template with stylish design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Placement Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 0;
        }
        .main-container {
            max-width: 900px;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            padding: 40px;
            margin: auto;
        }
        .header-title {
            color: #2d3748;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }
        .header-subtitle {
            color: #718096;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-label {
            font-weight: 600;
            color: #4a5568;
            font-size: 0.9rem;
        }
        .form-control, .form-select {
            border-radius: 8px;
            border: 1px solid #cbd5e0;
            padding: 10px 15px;
        }
        .form-control:focus, .form-select:focus {
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25);
            border-color: #667eea;
        }
        .btn-predict {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            transition: all 0.3s ease;
        }
        .btn-predict:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        .result-card {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 700;
        }
        .result-success {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .result-fail {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-container">
            <h2 class="header-title">🎓 Student Placement Predictor</h2>
            <p class="header-subtitle">Fill in the candidate academic details below to evaluate placement probability.</p>
            
            <form method="POST">
                <div class="row g-3">
                    <div class="col-md-4">
                        <label class="form-label">Serial Number (sl_no)</label>
                        <input type="number" class="form-control" name="sl_no" value="1" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Gender</label>
                        <select class="form-select" name="gender">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">10th Percentage (ssc_p)</label>
                        <input type="number" step="0.01" class="form-control" name="ssc_p" value="75.0" required>
                    </div>

                    <div class="col-md-4">
                        <label class="form-label">10th Board (ssc_b)</label>
                        <select class="form-select" name="ssc_b">
                            <option value="Central">Central</option>
                            <option value="Others">Others</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">12th Percentage (hsc_p)</label>
                        <input type="number" step="0.01" class="form-control" name="hsc_p" value="70.0" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">12th Board (hsc_b)</label>
                        <select class="form-select" name="hsc_b">
                            <option value="Central">Central</option>
                            <option value="Others">Others</option>
                        </select>
                    </div>

                    <div class="col-md-4">
                        <label class="form-label">12th Stream (hsc_s)</label>
                        <select class="form-select" name="hsc_s">
                            <option value="Commerce">Commerce</option>
                            <option value="Science">Science</option>
                            <option value="Arts">Arts</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Degree Percentage (degree_p)</label>
                        <input type="number" step="0.01" class="form-control" name="degree_p" value="68.0" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Degree Type (degree_t)</label>
                        <select class="form-select" name="degree_t">
                            <option value="Comm&Mgmt">Comm&Mgmt</option>
                            <option value="Sci&Tech">Sci&Tech</option>
                            <option value="Others">Others</option>
                        </select>
                    </div>

                    <div class="col-md-4">
                        <label class="form-label">Work Experience (workex)</label>
                        <select class="form-select" name="workex">
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Employability Test % (etest_p)</label>
                        <input type="number" step="0.01" class="form-control" name="etest_p" value="65.0" required>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">MBA Specialization</label>
                        <select class="form-select" name="specialisation">
                            <option value="Mkt&Fin">Mkt&Fin</option>
                            <option value="Mkt&HR">Mkt&HR</option>
                        </select>
                    </div>

                    <div class="col-md-12">
                        <label class="form-label">MBA Percentage (mba_p)</label>
                        <input type="number" step="0.01" class="form-control" name="mba_p" value="62.0" required>
                    </div>

                    <div class="col-md-12 mt-4">
                        <button type="submit" class="btn btn-predict">🚀 Predict Placement Status</button>
                    </div>
                </div>
            </form>

            {% if prediction %}
                {% if prediction|lower in ['1', 'placed', 'yes'] %}
                    <div class="result-card result-success">
                        🎉 Prediction Result: Candidate is <strong>PLACED</strong>!
                    </div>
                {% else %}
                    <div class="result-card result-fail">
                        ⚠️ Prediction Result: Candidate is <strong>NOT PLACED</strong>.
                    </div>
                {% endif %}
            {% endif %}

            {% if error %}
                <div class="alert alert-danger mt-4 text-center" role="alert">
                    {{ error }}
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    error_message = None

    if request.method == "POST":
        if model is None:
            error_message = "Model file 'Decision.pkl' not found on the server."
        else:
            try:
                # Extract form fields safely
                sl_no = int(request.form.get("sl_no", 1))
                gender = request.form.get("gender", "Male")
                ssc_p = float(request.form.get("ssc_p", 0.0))
                ssc_b = request.form.get("ssc_b", "Central")
                hsc_p = float(request.form.get("hsc_p", 0.0))
                hsc_b = request.form.get("hsc_b", "Central")
                hsc_s = request.form.get("hsc_s", "Commerce")
                degree_p = float(request.form.get("degree_p", 0.0))
                degree_t = request.form.get("degree_t", "Comm&Mgmt")
                workex = request.form.get("workex", "No")
                etest_p = float(request.form.get("etest_p", 0.0))
                specialisation = request.form.get("specialisation", "Mkt&Fin")
                mba_p = float(request.form.get("mba_p", 0.0))

                # Explicitly map categories to numeric values to prevent type conversion errors
                gender_val = 1 if gender == "Male" else 0
                ssc_b_val = 1 if ssc_b == "Central" else 0
                hsc_b_val = 1 if hsc_b == "Central" else 0
                hsc_s_val = {'Commerce': 0, 'Science': 1, 'Arts': 2}.get(hsc_s, 0)
                degree_t_val = {'Comm&Mgmt': 0, 'Sci&Tech': 1, 'Others': 2}.get(degree_t, 0)
                workex_val = 1 if workex == "Yes" else 0
                specialisation_val = 0 if specialisation == "Mkt&Fin" else 1

                # Construct input DataFrame with exact expected feature names and numeric types
                input_data = pd.DataFrame([{
                    'sl_no': sl_no,
                    'gender': gender_val,
                    'ssc_p': ssc_p,
                    'ssc_b': ssc_b_val,
                    'hsc_p': hsc_p,
                    'hsc_b': hsc_b_val,
                    'hsc_s': hsc_s_val,
                    'degree_p': degree_p,
                    'degree_t': degree_t_val,
                    'workex': workex_val,
                    'etest_p': etest_p,
                    'specialisation': specialisation_val,
                    'mba_p': mba_p
                }])

                # Make prediction
                pred = model.predict(input_data)[0]
                prediction_result = str(pred)

            except Exception as e:
                error_message = f"An error occurred during prediction: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction_result, error=error_message)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)