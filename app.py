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
    form_values = {field["name"]: "" for field in FIELD_CONFIG}

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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
