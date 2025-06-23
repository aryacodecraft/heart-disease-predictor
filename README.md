# 🫀 Heart Disease Predictor using K-Nearest Neighbors (KNN)

This project is a machine learning model that predicts the likelihood of a person having heart disease based on real medical attributes. It uses the **K-Nearest Neighbors (KNN)** algorithm to make predictions from a dataset collected from the **UCI Machine Learning Repository**.

> ⚠️ Disclaimer: This tool is for educational purposes only. It is NOT a real diagnostic tool.

---

## 🚀 Features

- 🧠 Built with Scikit-Learn’s `KNeighborsClassifier`
- 📊 Uses real-world data (`processed.cleveland.data`) from UCI
- 🔢 Takes user input from the console
- 📈 Predicts heart disease presence based on 13 medical factors
- 💬 Human-readable results ("You might have heart disease" or "You're good!")

---

## 📂 Project Structure

```bash
heart-disease-predictor/
├── src/
│   └── heart_knn_predictor.py      # Main code file
├── data/
│   └── processed.cleveland.data    # Dataset from UCI
├── README.md
├── requirements.txt
└── .gitignore


## Why K-Nearest Neighbors (KNN)?

KNN is a simple, yet powerful machine learning algorithm that works by:
- Looking at the 'k' closest data points (neighbors) to a new input.
- Voting based on their labels (in our case, whether they had heart disease or not).
- Predicting the majority class.

### Why KNN for this project?
- ✅ Works well with small- to medium-sized datasets
- ✅ No assumptions about data distribution (unlike logistic regression, etc.)
- ✅ Easy to explain to non-ML folks
- ✅ It’s non-parametric — meaning it adapts to the actual data pattern
- ✅ Can improve with data scaling (which we did with StandardScaler)

💡 We used k=5, a common starting point, to balance bias and variance.

## Input Features Used

| Feature Name         | Description                                             |
|----------------------|--------------------------------------------------------|
| age                  | Age of the person                                      |
| sex                  | 1 = male, 0 = female                                   |
| chest_pain_type      | 0 = typical angina, 1–3 = varying severity             |
| resting_bp           | Resting blood pressure                                 |
| cholesterol          | Serum cholesterol in mg/dl                             |
| fasting_blood_sugar  | >120 mg/dl (1 = true, 0 = false)                       |
| rest_ecg             | ECG results (0 = normal, 1/2 = abnormal)               |
| max_heart_rate       | Maximum heart rate achieved                            |
| exercise_angina      | 1 = yes, 0 = no                                        |
| oldpeak              | ST depression induced by exercise                      |
| slope                | Slope of ST segment during exercise                    |
| num_major_vessels    | Number of vessels shown in fluoroscopy                 |
| thalassemia          | 3 = normal, 6 = fixed defect, 7 = reversible defect    |

## Sample Output

```
Age: 58
Sex: 1
Chest pain type: 2
Resting BP: 130
Cholesterol: 250
Fasting blood sugar > 120? (1 = yes, 0 = no): 0
Resting ECG: 1
Max heart rate achieved: 160
Exercise angina? (1 = yes, 0 = no): 0
Oldpeak: 1.4
Slope: 1
Number of vessels: 0
Thalassemia: 3

⚠️  You might have heart disease. Please consult a doctor.
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/heart-disease-predictor.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script or notebook to train the model and make predictions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file