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

# 🚀 Day 4 Progress

## 🔹 Burnout Severity Monitoring System

Implemented a human-readable burnout severity analysis system using behavioral risk scoring.

### Severity Levels

* Low Risk
* Moderate Risk
* High Risk

The system now behaves more like a monitoring and decision-support application rather than only a machine learning classifier.

---

## 🔹 Severity-Based Recommendation Engine

Upgraded static recommendations into dynamic intervention plans based on severity level.

### Example Actions

* Reduce study overload
* Improve sleep schedule
* Schedule regular breaks
* Maintain balanced routines
* Reduce unnecessary screen time

This improved the practical usability of the project significantly.

---

## 🔹 SHAP Explainability Integration

Integrated SHAP (SHapley Additive Explanations) for advanced model interpretability.

### Added:

* SHAP feature importance visualization
* Model transparency
* Feature impact analysis

This helps explain:

* why predictions happen
* which features influence burnout most
* how behavioral indicators affect model output

---

## 🔹 Visualization & Portfolio Improvements

Added automatic saving of all generated plots into a dedicated visuals folder.

### Saved Visualizations

* SHAP Feature Importance
* Burnout Severity Distribution
* Feature Correlation Heatmap
* Mental Health Distribution
* Model Accuracy Comparison
* Screen Time Trend Analysis
* Feature Importance Rankings

---

## 🔹 Key Learning

Day 4 focused less on chasing accuracy and more on:

* explainability
* system intelligence
* decision support
* behavioral interpretation
* AI transparency

This reflects a more realistic real-world machine learning workflow.

# 🚀 Day 5 Progress

## 🔹 Interactive Prediction System

Built a reusable burnout prediction function capable of analyzing custom student behavioral data.

### Added Features

* Real-time burnout prediction
* Confidence score estimation
* Reusable inference pipeline
* Custom student profile support

The system can now predict burnout levels for new students instead of relying only on dataset samples.

---

## 🔹 Interactive CLI Demo

Implemented a demo-based prediction workflow for showcasing real-world usage.

### Demo Workflow

* Custom student data input
* Automated prediction generation
* Confidence analysis
* Instant burnout evaluation

This significantly improved the usability and presentation quality of the project.

---

## 🔹 Burnout Risk Report Generator

Added a human-readable burnout reporting system.

### Report Features

* Burnout level detection
* Confidence interpretation
* Severity analysis
* Risk alert generation

The project now behaves more like an AI-powered decision-support system instead of a basic machine learning script.

---

## 🔹 Advanced Explainability

Enhanced interpretability using SHAP Explainable AI.

### SHAP Improvements

* Global feature contribution analysis
* Feature importance interpretation
* Visual explanation of predictions
* Improved model transparency

This upgrade makes the system more aligned with real-world AI monitoring pipelines.

---

## 🔹 Visualization Improvements

Improved portfolio and GitHub presentation quality by automatically saving all visual outputs.

### Saved Visualizations

* Model Accuracy Comparison
* SHAP Feature Importance
* Mental Health Distribution
* Screen Time Trend
* Weekly Burnout Trend
* Correlation Heatmap
* Burnout Severity Distribution

---

## 🔹 System Evolution

The project evolved from:

```text
Machine Learning Classification Script
```

into:

```text
AI-Powered Student Burnout Monitoring & Recommendation System
```

---

## 🔹 Key Learning

> High-quality AI systems are not built only on prediction accuracy.
> Real-world value comes from interpretability, usability, explainability, monitoring, and actionable recommendations.




# 🚀 Day 6 Progress

## 🔹 Interactive Console-Based Prediction System
- Built a real-time user input system
- Added validated console interaction
- Implemented reusable prediction workflow
- Added intelligent input handling with error prevention

## 🔹 Automated Burnout Report Generator
- Added text-based report export system
- Reports now save automatically inside `/reports`
- Generated recommendations based on burnout severity

## 🔹 Improved System Architecture
- Refactored project into modular sections:
  - Prediction Engine
  - Visualization System
  - Report Generator
  - Input Validation Layer
  - Explainability Module

## 🔹 GitHub Professionalization
- Organized project into production-style folders:
  - `/models`
  - `/visuals`
  - `/reports`
- Added cleaner execution flow
- Improved code readability and maintainability

## 🔹 Enhanced Data Visualization
- Improved SHAP explainability visualization
- Added trend tracking visualization
- Added cleaner burnout severity analysis

## ⚙ System Workflow

Student Data  
↓  
Preprocessing  
↓  
Feature Engineering  
↓  
ML Prediction  
↓  
Risk Severity Analysis  
↓  
SHAP Explainability  
↓  
Personalized Recommendations  
↓  
Final Burnout Report  



## 🔹 Key Learning

> Real-world AI systems are not only about prediction accuracy.  
> A complete AI solution also requires:
> - user interaction
> - interpretability
> - modular architecture
> - reporting systems
> - actionable recommendations





🚀 Day 7 — Professional UI & Presentation Upgrade

🎯 Objective

Focused on improving software presentation, terminal experience, and overall project professionalism without adding unnecessary ML complexity.

The goal of Day 7 was to transform the project from a machine learning backend into a presentation-ready AI software system.

✅ Major Improvements
🎨 Professional System Banner

Added a startup-style terminal banner before model execution.

Features Displayed:
Burnout Prediction
SHAP Explainability
Personalized Recommendations
Interactive Student Analysis
Report Generation
Visualization Dashboard
Result:

The project now starts with a professional software-style introduction instead of raw console logs.

🚦 Colorized Burnout Risk Levels

Implemented a dedicated risk display system using emoji-based severity indicators.

Risk Labels:
🔴 HIGH RISK
🟠 MODERATE RISK
🟢 LOW RISK
Improvement:

Prediction outputs became more readable, user-friendly, and visually professional during live demonstrations.

📊 Clean Training Summary

Replaced raw accuracy logs with a cleaner training summary interface.

Before:
Gradient Boosting : 0.8667
After:
MODEL TRAINING COMPLETED
────────────────────────
Best Model : Gradient Boosting
Accuracy   : 86.67%
Result:

Improved readability and software presentation quality.

🧹 Codebase Cleanup

Performed structural cleanup and removed redundant outputs.

Improvements:
Removed duplicate comment sections
Replaced generic exception handling with except ValueError
Removed outdated model-save console logs
Simplified final execution flow
Improved terminal formatting consistency
📋 System Summary Dashboard

Added a professional end-screen system summary.

Includes:
Active prediction engine status
Explainable AI availability
Visualization generation status
Final model accuracy
System readiness confirmation
Result:

The project now ends like a deployable AI analytics product instead of abruptly terminating.

📈 Final Project Capabilities
✔ Multi-Model Machine Learning
✔ Feature Engineering
✔ SHAP Explainable AI
✔ Burnout Risk Prediction
✔ Interactive User Input System
✔ Personalized Recommendations
✔ TXT Report Generation
✔ Visualization Dashboard
✔ Professional Terminal UI
🏆 Best Performing Model
Model	Accuracy
Logistic Regression	80.37%
Random Forest	86.65%
Gradient Boosting	86.67%
Final Selected Model:

Gradient Boosting Classifier

✅ Current Project Status
SYSTEM READY FOR DEMONSTRATION 


## 👨‍💻 Author

Pranav K
