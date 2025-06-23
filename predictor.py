import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

column_names = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "slope",
    "num_major_vessels",
    "thalassemia",
    "target"
]

data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

updated_targets = []
for value in data["target"]:
    if int(value) == 0:
        updated_targets.append(0)
    else:
        updated_targets.append(1)

data["target"] = updated_targets

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


print("Enter the following patient details to predict heart disease:")

user_input = []
user_input.append(float(input("Age: ")))
user_input.append(int(input("Sex (1 = male, 0 = female): ")))
user_input.append(int(input("Chest pain type (0–3): ")))
user_input.append(float(input("Resting BP : ")))
user_input.append(float(input("Cholesterol : ")))
user_input.append(int(input("Fasting blood sugar > 120? (1 = yes, 0 = no): ")))
user_input.append(int(input("Resting ECG (0, 1, 2): ")))
user_input.append(float(input("Max heart rate achieved: ")))
user_input.append(int(input("Exercise-induced angina? (1 = yes, 0 = no): ")))
user_input.append(float(input("Oldpeak (ST depression): ")))
user_input.append(int(input("Slope (0–2): ")))
user_input.append(int(input("Number of major vessels (0–3): ")))
user_input.append(int(input("Thalassemia (3 = normal, 6 = fixed, 7 = reversible): ")))

user_df = pd.DataFrame([user_input], columns=X.columns)
user_scaled = scaler.transform(user_df)    #--> Standardizing input

prediction = knn.predict(user_scaled)

if prediction[0] == 1:
    print("You might have heart disease.")
else:
    print("You are unlikely to have heart disease.")
