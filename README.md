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


  # 🚀 Day 3 Progress

## 🎯 Day 3 Goals
Focused on improving:
- Data quality
- Feature engineering
- Behavioral analysis
- Explainability
- Recommendation system
- Visualization pipeline

Instead of blindly trying more models, the focus was placed on:
- smarter features
- reducing noise
- improving signal quality
- creating meaningful behavioral insights

---

# 🧠 Feature Engineering Improvements

Added custom behavioral intelligence features:

### ✅ Mental Health Score
Combines:
- anxiety score
- depression score
- academic pressure score

```python
mental_health_score =
    anxiety_score +
    depression_score +
    academic_pressure_score
```

### ✅ Study-to-Sleep Ratio
Measures overwork behavior:

```python
study_sleep_ratio =
    daily_study_hours /
    (daily_sleep_hours + 1)
```

### ✅ Wellness Score
Represents healthier lifestyle balance:

```python
wellness_score =
    physical_activity_hours +
    social_support_score -
    stress_level
```

These engineered features improved behavioral interpretation and made the dataset more meaningful.

---

# 🔍 Correlation Analysis

Added feature correlation analysis using:
- correlation matrix
- heatmap visualization

This helped identify:
- useful features
- weak/noisy features
- behavioral relationships

---

# 🤖 Improved ML Pipeline

### Models Used
- Logistic Regression
- Random Forest Classifier

### Pipeline Improvements
- Better preprocessing
- Feature scaling
- Feature engineering
- Cleaner modular code structure
- Balanced model training
- Model comparison workflow

---

# 🧠 Explainability System

The system now explains WHY a student may be at burnout risk.

### Example Risk Factors
- High screen time
- Poor sleep schedule
- High stress level
- Low wellness score

This improves:
- interpretability
- usability
- real-world practicality

---

# 💡 Recommendation Engine

The project now provides personalized recommendations such as:

- Reduce screen time
- Improve sleep cycle
- Increase physical activity
- Practice stress management

The system now:
1. predicts burnout
2. explains the causes
3. suggests improvements

---

# 📈 Confidence-Based Prediction

Added probability confidence scoring using:

```python
predict_proba()
```

### Example Output

```text
Prediction : Medium
Confidence : 33.78%
Risk Score : 13.80
```

---

# 📊 Visualizations Added

## 🔹 Feature Correlation Heatmap

![Correlation Heatmap](visuals/correlation_heatmap.png)

---

## 🔹 Mental Health Distribution

![Mental Health Distribution](visuals/mental_health_distribution.png)

---

## 🔹 Top Burnout Indicators

![Top Burnout Indicators](visuals/top_burnout_indicators.png)

---

## 🔹 Mental Health vs Screen Time

![Mental Health vs Screen Time](visuals/mental_health_vs_screen_time.png)

---

# 🧠 Key Learning from Day 3

This phase demonstrated that:

- Better feature engineering is often more important than trying multiple models
- Behavioral intelligence improves interpretability
- Explainability makes ML systems more useful
- Recommendation systems improve real-world usability
- Many burnout datasets are partially synthetic or weakly correlated

Therefore, the project focuses heavily on:
- feature engineering
- behavioral analysis
- explainability
- recommendation systems

instead of blindly optimizing accuracy.

---

# 🚀 Day 3 Achievements

✅ Feature Engineering  
✅ Correlation Analysis  
✅ Improved ML Pipeline  
✅ Confidence-Based Predictions  
✅ Explainability System  
✅ Recommendation Engine  
✅ Visualization Pipeline  
✅ Cleaner Modular Code Structure  
✅ Model Exporting with Joblib  
✅ Feature Importance Analysis  

## 👨‍💻 Author

Pranav K
