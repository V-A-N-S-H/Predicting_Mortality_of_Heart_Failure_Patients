from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart_failure_clinical_records_dataset.csv"

FIELD_CONFIG = [
    {"name": "age", "label": "Age (years)", "step": "0.1", "min": "0", "max": None, "help": "Patient age."},
    {"name": "anaemia", "label": "Anaemia", "step": "1", "min": "0", "max": "1", "help": "0 = No, 1 = Yes."},
    {
        "name": "creatinine_phosphokinase",
        "label": "Creatinine Phosphokinase (mcg/L)",
        "step": "1",
        "min": "0",
        "max": None,
        "help": "CPK enzyme level in blood.",
    },
    {"name": "diabetes", "label": "Diabetes", "step": "1", "min": "0", "max": "1", "help": "0 = No, 1 = Yes."},
    {
        "name": "ejection_fraction",
        "label": "Ejection Fraction (%)",
        "step": "1",
        "min": "0",
        "max": "100",
        "help": "Percentage of blood leaving the heart each contraction.",
    },
    {
        "name": "high_blood_pressure",
        "label": "High Blood Pressure",
        "step": "1",
        "min": "0",
        "max": "1",
        "help": "0 = No, 1 = Yes.",
    },
    {
        "name": "platelets",
        "label": "Platelets (kiloplatelets/mL)",
        "step": "0.1",
        "min": "0",
        "max": None,
        "help": "Platelet count.",
    },
    {
        "name": "serum_creatinine",
        "label": "Serum Creatinine (mg/dL)",
        "step": "0.01",
        "min": "0",
        "max": None,
        "help": "Creatinine level in blood.",
    },
    {
        "name": "serum_sodium",
        "label": "Serum Sodium (mEq/L)",
        "step": "1",
        "min": "0",
        "max": None,
        "help": "Sodium level in blood.",
    },
    {"name": "sex", "label": "Sex", "step": "1", "min": "0", "max": "1", "help": "0 = Female, 1 = Male."},
    {"name": "smoking", "label": "Smoking", "step": "1", "min": "0", "max": "1", "help": "0 = No, 1 = Yes."},
    {
        "name": "time",
        "label": "Follow-up Time (days)",
        "step": "1",
        "min": "0",
        "max": None,
        "help": "Follow-up period in days.",
    },
]

FEATURES = [field["name"] for field in FIELD_CONFIG]
BINARY_FIELDS = {"anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"}


def train_model():
    data = pd.read_csv(DATA_PATH)

    x = data[FEATURES]
    y = data["DEATH_EVENT"]

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = SVC(probability=True, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model, scaler


model, scaler = train_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None
    form_values = {
        "age": "60",
        "anaemia": "0",
        "creatinine_phosphokinase": "250",
        "diabetes": "0",
        "ejection_fraction": "38",
        "high_blood_pressure": "0",
        "platelets": "263000",
        "serum_creatinine": "1.1",
        "serum_sodium": "137",
        "sex": "1",
        "smoking": "0",
        "time": "130",
    }

    if request.method == "POST":
        try:
            input_values = []

            for field in FIELD_CONFIG:
                field_name = field["name"]
                raw_value = request.form.get(field_name, "").strip()

                if raw_value == "":
                    raise ValueError(f"{field['label']} is required.")

                value = float(raw_value)
                form_values[field_name] = raw_value

                if field_name in BINARY_FIELDS and value not in (0.0, 1.0):
                    raise ValueError(f"{field['label']} must be 0 or 1.")

                input_values.append(value)

            input_df = pd.DataFrame([input_values], columns=FEATURES)
            input_scaled = scaler.transform(input_df)

            predicted_class = int(model.predict(input_scaled)[0])
            predicted_probability = float(model.predict_proba(input_scaled)[0][1]) * 100

            if predicted_class == 1:
                prediction = "High Mortality Risk"
            else:
                prediction = "Lower Mortality Risk"

            probability = round(predicted_probability, 2)

        except ValueError as exc:
            error = str(exc)

    return render_template(
        "index.html",
        fields=FIELD_CONFIG,
        form_values=form_values,
        prediction=prediction,
        probability=probability,
        error=error,
    )

@app.route("/api/dashboard-data")
def dashboard_data():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Age Distribution (bins)
    age_bins = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 80, 100], labels=['<40', '40-50', '50-60', '60-70', '70-80', '80+'])
    age_dist = df.groupby(age_bins, observed=False)['DEATH_EVENT'].value_counts().unstack().fillna(0)
    age_dist_data = {
        "labels": age_dist.index.tolist(),
        "survived": age_dist[0].tolist(),
        "died": age_dist[1].tolist()
    }
    
    # 2. Gender and Mortality
    sex_dist = df.groupby('sex')['DEATH_EVENT'].value_counts().unstack().fillna(0)
    sex_dist_data = {
        "labels": ["Female", "Male"],
        "survived": [int(sex_dist.loc[0, 0]), int(sex_dist.loc[1, 0])],
        "died": [int(sex_dist.loc[0, 1]), int(sex_dist.loc[1, 1])]
    }
    
    # 3. Ejection Fraction vs Serum Creatinine (Scatter)
    scatter_survived = df[df['DEATH_EVENT'] == 0][['ejection_fraction', 'serum_creatinine']].to_dict(orient='records')
    scatter_died = df[df['DEATH_EVENT'] == 1][['ejection_fraction', 'serum_creatinine']].to_dict(orient='records')
    scatter_data = {
        "survived": scatter_survived,
        "died": scatter_died
    }
    
    # 4. Binary Features Pie Chart (overall prevalence)
    binary_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']
    binary_prevalence = [int(df[feat].sum()) for feat in binary_features]
    binary_data = {
        "labels": ["Anaemia", "Diabetes", "High Blood Pressure", "Smoking"],
        "data": binary_prevalence
    }

    return {
        "ageDistribution": age_dist_data,
        "sexDistribution": sex_dist_data,
        "ejectionVsCreatinine": scatter_data,
        "binaryPrevalence": binary_data,
        "totalRecords": len(df),
        "mortalityRate": round((df['DEATH_EVENT'].sum() / len(df)) * 100, 1)
    }




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
