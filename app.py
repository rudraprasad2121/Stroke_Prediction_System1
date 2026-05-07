import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# ---------------- APP ----------------
app = Flask(__name__)
app.secret_key = "welcome"

# ---------------- LOAD & TRAIN ONCE ----------------
dataset = pd.read_csv("Dataset/healthcare-dataset-stroke-data.csv")
dataset.fillna(0, inplace=True)

# Encoders
encoders = {}
for col in ['gender','ever_married','work_type','Residence_type','smoking_status']:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))
    encoders[col] = le

# Split data
Y = dataset['stroke']
X = dataset.drop(['id','stroke'], axis=1)

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection
selector = SelectKBest(score_func=chi2, k=9)
X_selected = selector.fit_transform(X_scaled, Y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        # -------- GET INPUT --------
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart = int(request.form['heart'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoke = request.form['smoke']

        # -------- ENCODE --------
        gender = encoders['gender'].transform([gender])[0]
        married = encoders['ever_married'].transform([married])[0]
        work = encoders['work_type'].transform([work])[0]
        residence = encoders['Residence_type'].transform([residence])[0]
        smoke = encoders['smoking_status'].transform([smoke])[0]

        # -------- CREATE INPUT --------
        input_data = np.array([[gender, age, hypertension, heart,
                                married, work, residence,
                                glucose, bmi, smoke]])

        # -------- SCALE + SELECT --------
        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)

        # -------- PREDICT --------
        prob = model.predict_proba(input_selected)[0][1]

        # -------- SMART RISK LOGIC --------
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        result = "Stroke" if prob > 0.5 else "Normal"

        # -------- SAFETY RULE (IMPORTANT) --------
        if glucose < 140 and bmi < 30 and hypertension == 0:
            result = "Normal"
            risk = "Low"
            prob = min(prob, 0.3)

        return render_template("result.html",
                               result=result,
                               prob=round(prob*100,2),
                               risk=risk)

    except Exception as e:
        return str(e)


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)