import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import callbacks

from sklearn.metrics import classification_report

# Load Dataset
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Features and Target
X = data.drop("DEATH_EVENT", axis=1)
Y = data["DEATH_EVENT"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# ---------------- SVM MODEL ---------------- #

svm_model = svm.SVC()

svm_model.fit(X_train, Y_train)

svm_prediction = svm_model.predict(X_test)

print("SVM Classification Report")
print(classification_report(Y_test, svm_prediction))

# ---------------- ANN MODEL ---------------- #

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

ann_model = Sequential()

ann_model.add(Dense(units=16, activation="relu", input_dim=12))
ann_model.add(Dense(units=8, activation="relu"))
ann_model.add(Dropout(0.25))

ann_model.add(Dense(units=8, activation="relu"))
ann_model.add(Dropout(0.5))

ann_model.add(Dense(units=1, activation="sigmoid"))

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ann_model.fit(
    X_train,
    Y_train,
    batch_size=25,
    epochs=100,
    validation_split=0.25,
    callbacks=[early_stopping]
)

# Prediction
ann_prediction = ann_model.predict(X_test)
ann_prediction = (ann_prediction > 0.5).astype(int)

print("ANN Classification Report")
print(classification_report(Y_test, ann_prediction))