# 🔥 Student Burnout Prediction & Recommendation System

## 📌 Overview

This project focuses on predicting student burnout levels using machine learning techniques based on academic, behavioral, and lifestyle data. The goal is to identify students at risk and provide future recommendations to improve their mental well-being.

---

## 🎯 Objectives

* Predict burnout levels (Low, Medium, High)
* Analyze factors affecting student mental health
* Build a foundation for a recommendation system

---

## 📊 Dataset Features

The dataset includes:

* Academic data (CGPA, attendance, study hours)
* Mental health indicators (stress, anxiety, depression scores)
* Lifestyle factors (sleep, screen time, physical activity)
* Social factors (support score, financial stress)

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn

---

## 🚀 Day 1 Progress

### ✔️ Data Preprocessing

* Cleaned column names
* Converted categorical target (`burnout_level`) into numeric values
* Removed unnecessary columns (student_id)
* Applied One-Hot Encoding to categorical features

### ✔️ Model Building

* Implemented Logistic Regression as a baseline model
* Split dataset into training and testing sets (80-20)

### ✔️ Evaluation

* Used classification report to evaluate performance

---

## 📈 Results

* Accuracy: ~33%
* Observation:

  * Model performs similar to random guessing
  * Indicates dataset complexity and need for better models

---

## 🧠 Key Insight

This baseline model establishes a starting point. Future improvements will focus on:

* Advanced models (Random Forest, XGBoost)
* Feature engineering
* Handling data imbalance

---

## 🔮 Future Scope

* Build a recommendation system for burnout reduction
* Add visualization dashboard
* Deploy as a web application

---

## 📂 Project Structure

```
├── major_preprocessing.py
├── student_mental_health_burnout.csv
└── README.md
```

---


## 🚀 Day 2 Progress

### 🔹 Exploratory Data Analysis (EDA)
- Analyzed dataset structure using:
  - head()
  - info()
  - describe()
  - value_counts()

### 🔹 Preprocessing Improvements
- Removed unnecessary columns
- Applied ordinal encoding for ordered categorical features
- Applied one-hot encoding for nominal categorical features
- Added feature scaling using StandardScaler

### 🔹 Model Comparison
Implemented and compared:
- Logistic Regression
- Random Forest Classifier

### 🔹 Evaluation Metrics
Used:
- Accuracy Score
- Classification Report
- Confusion Matrix

### 🔹 Feature Importance Analysis
Identified key behavioral factors influencing burnout prediction:
- CGPA
- Attendance Percentage
- Screen Time
- Study Hours
- Sleep Hours

### 🔹 Key Learning
- Understood importance of preprocessing and feature engineering
- Learned differences between linear and ensemble models
- Identified limitations of synthetic burnout datasets

## 👨‍💻 Author

Pranav K
