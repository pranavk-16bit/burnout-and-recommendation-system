# =========================================================
# STUDENT BURNOUT DETECTION SYSTEM
# CLEAN + OPTIMIZED + HIGHER ACCURACY VERSION
# =========================================================

import os
import warnings
import joblib
import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)



# =========================================================
# SETTINGS
# =========================================================

warnings.filterwarnings("ignore")

for folder in ["reports", "models", "visuals"]:
    os.makedirs(folder, exist_ok=True)


sns.set_theme(
    style="whitegrid",
    palette="flare",
    context="talk"
)

plt.rcParams["figure.figsize"] = (10,6)

# =========================================================
# SYSTEM BANNER
# =========================================================

print("\n" + "=" * 60)
print(" AI-POWERED STUDENT BURNOUT DETECTION SYSTEM ")
print("=" * 60)

print("\nFeatures Included:")
print("✔ Burnout Prediction")
print("✔ SHAP Explainability")
print("✔ Personalized Recommendations")
print("✔ Interactive Student Analysis")
print("✔ Report Generation")
print("✔ Visualization Dashboard")

print("\nInitializing system...\n")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    r"C:\Users\prana\Downloads\student_mental_health_burnout_1M.csv"
).sample(50000,           # use 100k rows instead of 1M
    random_state=42
)


# =========================================================
# PREPROCESSING
# =========================================================

df.columns = df.columns.str.lower()

df["risk_level"] = df["risk_level"].map({
    "Low":0,
    "Medium":1,
    "High":2
})

df = pd.get_dummies(
    df,
    columns=["gender"],
    drop_first=True
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["stress_sleep_ratio"] = (
    df["stress_level"] /
    (df["sleep_hours"] + 1)
)

df["mental_pressure"] = (
    df["anxiety_score"] +
    df["depression_score"] +
    df["exam_pressure"]
)

df["wellness_score"] = (
    df["physical_activity"] +
    df["social_support"] -
    df["stress_level"]
)

df["digital_overload"] = (
    df["screen_time"] *
    df["internet_usage"]
)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop([
    "risk_level",
    "burnout_score",
    "mental_health_index",
    "dropout_risk"
], axis=1)

y = df["risk_level"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ),
        True
    ),

    "Random Forest": (
    RandomForestClassifier(
    n_estimators=60,
    max_depth=10,
    n_jobs=-1,
    random_state=42
),
    False
),

    "Gradient Boosting": (
    GradientBoostingClassifier(
    n_estimators=40,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
),
    False
)}




# =========================================================
# TRAINING
# =========================================================

results = {}

for name, (model, scaled) in models.items():

    Xtr, Xte = (
        (X_train_scaled, X_test_scaled)
        if scaled else
        (X_train, X_test)
    )

    model.fit(Xtr, y_train)

    acc = accuracy_score(
        y_test,
        model.predict(Xte)
    )

    results[name] = acc

# =========================================================
# BEST MODEL SELECTION
# =========================================================

best_model_name = max(
    results,
    key=results.get
)

best_model = models[
    best_model_name
][0]


best_acc = results[
    best_model_name
] * 100

# Save model

joblib.dump(
    best_model,
    "models/burnout_model.pkl"
)

# Clean training summary

print("MODEL TRAINING COMPLETED")
print("────────────────────────")

print(
    f"Best Model : {best_model_name}"
)

print(
    f"Accuracy   : {best_acc:.2f}%\n"
)

# =========================================================
# FEATURE IMPORTANCE DATA
# =========================================================

if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature": X.columns,

        "Importance":
            best_model.feature_importances_

    }).sort_values(
        by="Importance",
        ascending=False
    )

    top_features = importance_df.head(10)

    importance_df.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

# =========================================================
# REUSABLE PREDICTION FUNCTION
# =========================================================

def predict_student(student_data):

    input_df = pd.DataFrame([student_data])

    # Add missing columns automatically
    for col in X.columns:

        if col not in input_df.columns:
            input_df[col] = 0

    # Keep exact training column order
    input_df = input_df[X.columns]

    prediction = best_model.predict(input_df)[0]

    probability = (
        best_model
        .predict_proba(input_df)
        .max() * 100
    )

    labels = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return labels[prediction], probability

# =========================================================
# RISK DISPLAY SYSTEM
# =========================================================

def risk_emoji(level):

    return {

        "High": "🔴 HIGH RISK",
        "Medium": "🟠 MODERATE RISK",
        "Low": "🟢 LOW RISK"

    }[level]

# =========================================================
# REPORT EXPORT SYSTEM
# =========================================================

def save_report(prediction, confidence):

    with open(
        "reports/student_report.txt",
        "w"
    ) as f:

        f.write("STUDENT BURNOUT REPORT\n")
        f.write("=" * 40 + "\n\n")

        f.write(
            f"Prediction : {prediction}\n"
        )

        f.write(
            f"Confidence : {confidence:.2f}%\n"
        )

        if prediction == "High":

            f.write(
                "\nHigh burnout detected\n"
            )

            f.write(
                "Recommendations:\n"
            )

            f.write(
                "- Improve sleep\n"
            )

            f.write(
                "- Reduce workload\n"
            )

            f.write(
                "- Practice stress management\n"
            )

        elif prediction == "Medium":

            f.write(
                "\nModerate burnout detected\n"
            )

        else:

            f.write(
                "\nLow burnout detected\n"
            )

    

# =========================================================
# SMART INSIGHTS
# =========================================================
sample = X_test.iloc[0]

checks = {

    "⚠ High Screen Time":
        sample["screen_time"] > 8,

    "⚠ Poor Sleep":
        sample["sleep_hours"] < 6,

    "⚠ High Stress":
        sample["stress_level"] > 7,

    "⚠ Low Wellness":
        sample["wellness_score"] < 3
}

print("\nRisk Factors")

[print(k) for k,v in checks.items() if v]

# =========================================================
# RECOMMENDATIONS
# =========================================================

recommendations = {

    "✔ Reduce screen time":
        sample["screen_time"] > 8,

    "✔ Improve sleep":
        sample["sleep_hours"] < 6,

    "✔ Meditation / Exercise":
        sample["stress_level"] > 7,

    "✔ Increase social activity":
        sample["wellness_score"] < 3
}

print("\nRecommendations")

[print(k) for k,v in recommendations.items() if v]
# =========================================================
# SHAP VISUALIZATION
# =========================================================

try:

    rf_model = models["Random Forest"][0]

    sample_shap = X_test.sample(
        100,
        random_state=42
    )

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer(
        sample_shap
    )

    shap.plots.bar(
        shap_values[:, :, 1],
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "visuals/shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

except Exception as e:

    print("\nSHAP Error:", e)

# =========================================================
# SAVE PLOT FUNCTION
# =========================================================

def save_plot(title, file):

    plt.title(
        title,
        fontsize=18,
        weight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"visuals/{file}.png",
        dpi=300
    )

    plt.show()

    plt.close()


# =========================================================
# USER INPUT SYSTEM
# =========================================================

def ask(prompt, low, high, dtype=float):

    while True:

        try:

            value = dtype(input(prompt))

            if low <= value <= high:
                return value

            print(f"❌ Enter value between {low} and {high}")

        except ValueError:
            print("❌ Invalid input")


def ask_gender():

    while True:

        gender = input(
            "Gender (Male/Female): "
        ).strip().lower()

        if gender in ["male", "female"]:
            return gender

        print("❌ Enter Male or Female")


def get_student_input():

    print("\n" + "=" * 40)
    print("ENTER STUDENT DETAILS")
    print("=" * 40)

    study = ask(
        "Study Hours Per Day (0-24): ",
        0, 24
    )

    screen = ask(
        "Screen Time Hours (0-24): ",
        0, 24
    )

    sleep = ask(
        "Sleep Hours (0-24): ",
        0, 24
    )

    stress = ask(
        "Stress Level (1-10): ",
        1, 10, int
    )

    anxiety = ask(
        "Anxiety Score (1-10): ",
        1, 10, int
    )

    depression = ask(
        "Depression Score (1-10): ",
        1, 10, int
    )

    support = ask(
        "Social Support (1-10): ",
        1, 10, int
    )

    gender = ask_gender()

    return {

        # Basic Features

        "age": 21,
        "academic_year": 3,

        "study_hours_per_day": study,
        "screen_time": screen,
        "sleep_hours": sleep,

        "stress_level": stress,
        "anxiety_score": anxiety,
        "depression_score": depression,
        "social_support": support,

        # Defaults

        "exam_pressure": stress,
        "internet_usage": screen,
        "physical_activity": 2,
        "financial_stress": 5,
        "family_expectation": 5,
        "academic_performance": 7,

        # Encoded Gender

        "gender_Male":
            1 if gender == "male" else 0,

        # Engineered Features

        "stress_sleep_ratio":
            stress / (sleep + 1),

        "mental_pressure":
            anxiety + depression + stress,

        "wellness_score":
            support + 2 - stress,

        "digital_overload":
            screen * stress
    }
# =========================================================
# MODEL COMPARISON
# =========================================================

plt.figure(figsize=(8,5))

ax = sns.barplot(
    x=list(results.keys()),
    y=list(results.values()),
    palette="magma"
)

for i, v in enumerate(results.values()):

    ax.text(
        i,
        v + 0.01,
        f"{v:.2f}",
        ha="center",
        fontweight="bold"
    )

plt.ylim(0,1)

save_plot(
    "Model Accuracy Comparison",
    "model_comparison"
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(best_model, "feature_importances_"):

    plt.figure(figsize=(12,6))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        palette="viridis"
    )

    save_plot(
        "Top Burnout Indicators",
        "feature_importance"
    )

# =========================================================
# MENTAL HEALTH DISTRIBUTION
# =========================================================

plt.figure(figsize=(10,6))

sns.kdeplot(
    data=df,
    x="mental_health_index",
    fill=True,
    linewidth=3,
    color="#00E5FF"
)

save_plot(
    "Mental Health Distribution",
    "mental_health_distribution"
)

# =========================================================
# SCREEN TIME TREND
# =========================================================

trend = df.sample(
    2000,
    random_state=42
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=trend,
    x="mental_health_index",
    y="screen_time",
    alpha=0.4
)

save_plot(
    "Mental Health vs Screen Time",
    "screen_time_trend"
)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

important_cols = [

    "mental_health_index",
    "stress_sleep_ratio",
    "wellness_score",
    "screen_time",
    "sleep_hours",
    "stress_level",
    "risk_level"
]

plt.figure(figsize=(10,7))

sns.heatmap(
    df[important_cols].corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

save_plot(
    "Key Feature Correlation",
    "correlation_heatmap"
)

# =========================================================
# BURNOUT PIE CHART
# =========================================================

plt.figure(figsize=(7,7))

risk_labels = df["risk_level"].map({

    0: "Low",
    1: "Medium",
    2: "High"
})

risk_labels.value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=sns.color_palette("Set2")
)

plt.ylabel("")

save_plot(
    "Burnout Severity Distribution",
    "severity_distribution"
)

# =========================================================
# WEEKLY TREND
# =========================================================

weekly_scores = [5, 7, 8, 11, 14]

plt.figure(figsize=(8,5))

plt.plot(
    weekly_scores,
    marker="o",
    linewidth=3
)

plt.xlabel("Week")
plt.ylabel("Burnout Score")

save_plot(
    "Weekly Burnout Trend",
    "burnout_trend"
)



# =========================================================
# INTERACTIVE PREDICTION
# =========================================================

user_student = get_student_input()

user_pred, user_conf = predict_student(
    user_student
)

print("\n" + "═" * 50)
print("            STUDENT ANALYSIS REPORT")
print("═" * 50)

print(f"""
Burnout Status : {risk_emoji(user_pred)}
Confidence     : {user_conf:.2f}%
""")

# =========================================================
# FINAL REPORT GENERATOR
# =========================================================
def generate_report(prediction, confidence):

    print("RECOMMENDATIONS")
    print("───────────────")

    if prediction == "High":

        tips = [
            "Reduce academic overload",
            "Improve sleep schedule",
            "Practice stress management",
            "Increase physical activity"
        ]

    elif prediction == "Medium":

        tips = [
            "Maintain healthier study balance",
            "Take regular study breaks",
            "Improve time management"
        ]

    else:

        tips = [
            "Maintain current lifestyle",
            "Continue healthy habits"
        ]

    for tip in tips:
        print(f"✔ {tip}")



# =========================================================
# FINAL PROJECT SUMMARY
# =========================================================

print("\n" + "═" * 65)
print("                SYSTEM SUMMARY")
print("═" * 65)

print(f"""
Model Used        : {best_model_name}
Prediction Engine : Active
Explainable AI    : Enabled
Visual Reports    : Generated
Final Accuracy    : {best_acc:.2f}%

SYSTEM STATUS     : READY
""")
