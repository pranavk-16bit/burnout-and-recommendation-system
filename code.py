# =========================================================
# STUDENT BURNOUT DETECTION SYSTEM
# CLEAN + OPTIMIZED + HIGHER ACCURACY VERSION
# =========================================================

import os, warnings, joblib,numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from datetime import datetime

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

df["sleep_quality"] = (
    df["sleep_hours"] / (df["screen_time"] + 1)
)

df["stress_index"] = (
    df["stress_level"] *
    df["exam_pressure"]
)

df["lifestyle_balance"] = (
    df["physical_activity"] +
    df["social_support"] -
    df["screen_time"]
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

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)
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

),
    "XGBoost": (
    XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    ),
    False
)
}




# =========================================================
# TRAINING
# =========================================================

results = {}
f1_scores = {}

for name, (model, scaled) in models.items():

    Xtr, Xte = (
        (X_train_scaled, X_test_scaled)
        if scaled else
        (X_train, X_test)
    )

    model.fit(Xtr, y_train)

    preds = model.predict(Xte)

    acc = accuracy_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds,
        average="weighted"
    )

    results[name] = acc
    f1_scores[name] = f1


# =========================================================
# BEST MODEL SELECTION
# =========================================================

best_model_name = max(
    f1_scores,
    key=f1_scores.get
)

best_model = models[
    best_model_name
][0]
joblib.dump(
    best_model,
    "models/burnout_model.pkl"
)

# CROSS VALIDATION

scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5
)

print(
    f"Cross Validation Accuracy : "
    f"{scores.mean()*100:.2f}%"
)

best_acc = results[
    best_model_name
] * 100

# Save model

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/features.pkl"
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

predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print("─────────────────────")

print(
    classification_report(
        y_test,
        predictions
    )
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print(f"F1 Score : {f1:.4f}")
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

    probs = best_model.predict_proba(input_df)[0]

    labels = {
        0: "Low",
        1: "Medium",
        2: "High"
}

    return (
        labels[prediction],
        probs.max() * 100,
        probs
)

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
# HISTORY TRACKING
# =========================================================

def save_history(score, sleep_hours, prediction):
    history_file = (
        "reports/student_history.csv"
    )

    new_data = pd.DataFrame({

    "date":[
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ],

    "burnout_score":[score],

    "sleep_hours":[
        sleep_hours
    ],
    "risk_level":[prediction]

})

    if os.path.exists(history_file):

        old = pd.read_csv(
            history_file
        )

        updated = pd.concat(
            [old, new_data],
            ignore_index=True
        )

    else:

        updated = new_data

    updated.to_csv(
        history_file,
        index=False
    )

def burnout_trend():

    history = pd.read_csv(
        "reports/student_history.csv"
    )

    if len(history) < 2:
        return

    plt.figure(figsize=(12,6))

    plt.plot(
        history["date"],
        history["burnout_score"],
        marker="o",
        linewidth=4,
        markersize=10,
        color="darkorange"
    )

    plt.fill_between(
        history["date"],
        history["burnout_score"],
        alpha=0.25,
        color="orange"
    )

    plt.grid(alpha=0.3)

    plt.ylabel("Burnout Score")
    plt.xlabel("Date")

    plt.tight_layout()

    plt.savefig(
        "visuals/burnout_trend.png",
        dpi=300
    )

    plt.show()
    plt.close()


def sleep_burnout_trend():

    history = pd.read_csv(
        "reports/student_history.csv"
    )

    if len(history) < 2:
        return

    fig, ax1 = plt.subplots(
        figsize=(12,6)
    )

    ax1.plot(
        history["date"],
        history["burnout_score"],
        marker="o",
        linewidth=4,
        color="crimson",
        label="Burnout"
    )

    ax1.set_ylabel(
        "Burnout Score",
        color="crimson"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        history["date"],
        history["sleep_hours"],
        marker="s",
        linewidth=4,
        color="royalblue",
        label="Sleep"
    )

    ax2.set_ylabel(
        "Sleep Hours",
        color="royalblue"
    )

    plt.title(
        "Sleep vs Burnout Relationship",
        fontsize=16,
        weight="bold"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "visuals/sleep_burnout_trend.png",
        dpi=300
    )

    plt.show()
    plt.close()

def early_warning():

    history_file = (
        "reports/student_history.csv"
    )

    if not os.path.exists(
        history_file
    ):
        return

    history = pd.read_csv(
        history_file
    )

    if len(history) < 7:
        return

    current = history[
        "burnout_score"
    ].iloc[-1]

    weekly_avg = history[
        "burnout_score"
    ].tail(7).mean()

    increase = (

        (current - weekly_avg)

        /

        weekly_avg

    ) * 100

    if increase > 25 and current > 2:

        print(
            "\n🚨 ALERT"
    )

        print(
            "Burnout risk increasing rapidly."
    )

        print(
            "Immediate intervention recommended."
    )

        print(
            "\n⚠ EARLY WARNING"
        )

        print(

            f"Burnout increased "

            f"{increase:.1f}% "

            f"vs weekly average"

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

    mental_health = ask(
        "Mental Health Score (1-10): ",
        1,
        10,
        int
)
    stress = max(1, 11 - mental_health)

    anxiety = max(1, round((11 - mental_health) * 0.9))

    depression = max(1, round((11 - mental_health) * 0.8))

    support = mental_health
    
    activity = ask(
    "Physical Activity Hours (0-10): ",
    0, 10
    )
    mental_health_score = (
    mental_health
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
        "physical_activity": activity,
        "mental_health_score": mental_health_score,
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

def burnout_indicator(score):

    plt.figure(figsize=(8,2))

    plt.barh(
        ["Burnout Score"],
        [score]
    )

    plt.xlim(0,3)

    save_plot(
        "Burnout Indicator",
        "burnout_indicator"
    )

def burnout_dashboard(student):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15,6)
    )

    # =====================================
    # LEFT SIDE
    # TOP BURNOUT INDICATORS
    # =====================================

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=axes[0]
    )

    axes[0].set_title(
        "Top Burnout Indicators",
        fontsize=14,
        weight="bold"
    )

    # =====================================
    # RIGHT SIDE
    # PERSONAL PROFILE
    # =====================================

    profile = {

        "Mental Health":
            student["mental_health_score"],

        "Sleep":
            student["sleep_hours"],

        "Activity":
            student["physical_activity"],

        "Study":
            student["study_hours_per_day"],

        "Screen":
            student["screen_time"]
    }

    axes[1].pie(
    profile.values(),
    labels=profile.keys(),
    startangle=90,
    wedgeprops={
        "width":0.45
    }
)

    axes[1].set_title(
        "Personal Wellness Profile",
        fontsize=14,
        weight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        "visuals/burnout_dashboard.png",
        dpi=300
    )

    plt.show()

    plt.close()

def risk_factor_relationships():

    correlation = df[[
        "stress_level",
        "sleep_hours",
        "screen_time",
        "wellness_score",
        "risk_level"
    ]].corr()["risk_level"]

    correlation = correlation.drop(
        "risk_level"
    )

    correlation = correlation.sort_values()

    plt.figure(figsize=(10,5))

    colors = [
        "green" if x < 0 else "crimson"
        for x in correlation.values
    ]

    plt.barh(
        correlation.index,
        correlation.values,
        color=colors
    )

    plt.axvline(
        x=0,
        color="black",
        linewidth=1
    )

    plt.xlabel(
        "Correlation With Burnout Risk"
    )

    plt.title(
        "Risk Factor Relationships",
        fontsize=16,
        weight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        "visuals/risk_factor_relationships.png",
        dpi=300
    )

    plt.show()

    plt.close()
# =========================================================
# INTERACTIVE PREDICTION
# =========================================================

user_student = get_student_input()

user_pred, user_conf, probs = predict_student(
    user_student
)
burnout_score = (

    probs[0] * 1 +

    probs[1] * 2 +

    probs[2] * 3

)

wellness = (
    (
        user_student["sleep_hours"] +
        user_student["physical_activity"] +
        user_student["social_support"]
    ) / 30
) * 10

burnout_indicator(
    burnout_score
)
burnout_dashboard(
    user_student
)
risk_factor_relationships()

save_history(
    burnout_score,
    user_student["sleep_hours"],
    user_pred
)
burnout_trend()

sleep_burnout_trend()


early_warning()

def burnout_status(score):

    if score < 1.5:

        return "🟢 Stable"

    elif score < 2.3:

        return "🟠 Watchlist"

    else:

        return "🔴 Critical"

print("\n" + "═" * 50)
print("            STUDENT ANALYSIS REPORT")
print("═" * 50)

print(f"""
Burnout Status : {risk_emoji(user_pred)}
Confidence     : {user_conf:.2f}%
Low Risk       : {probs[0]*100:.2f}%
Medium Risk    : {probs[1]*100:.2f}%
High Risk      : {probs[2]*100:.2f}%
Burnout Score  : {burnout_score:.2f}
Wellness Score : {wellness:.1f}/10
Monitoring     : {burnout_status(burnout_score)}
""")


# =====================================================
# EXPLAINABLE AI
# =====================================================

feature_names = {

    "stress_sleep_ratio":
        "Stress vs Sleep Balance",

    "stress_level":
        "Stress Level",

    "mental_pressure":
        "Mental Pressure",

    "wellness_score":
        "Wellness Score",

    "screen_time":
        "Screen Time",

    "sleep_hours":
        "Sleep Hours",

    "social_support":
        "Social Support",

    "depression_score":
        "Depression Score",

    "anxiety_score":
        "Anxiety Score",

    "gender_Male": "Gender",
    "academic_year": "Academic Year",
    "digital_overload": "Digital Overload",
    "stress_index": "Stress Index"
}

print("\nTOP REASONS FOR PREDICTION")
print("──────────────────────────")

for i, row in enumerate(
    top_features.head(3).itertuples(),
    start=1
):
    print(
        f"{i}. {feature_names.get(row.Feature, row.Feature)}"
    )

print("\nMONITORING STATUS")
print("────────────────")

history = pd.read_csv(
    "reports/student_history.csv"
)

history["risk_level"] = (
    history["risk_level"]
    .fillna("Unknown")
)

if len(history) >= 3:

    latest = history["burnout_score"].iloc[-1]

    previous = history["burnout_score"].iloc[-2]

    if latest > previous:

        print(
            "⚠ Burnout trend increasing"
        )

    elif latest < previous:

        print(
            "✅ Burnout improving"
        )

    else:

        print(
            "➖ Stable"
        )

print("\nRECENT MONITORING HISTORY")
print("────────────────────────")

history["risk_level"] = (
    history["risk_level"]
    .fillna("Unknown")
)

print(
    history[
        ["date", "burnout_score", "risk_level"]
    ].tail(5)
)

# =========================================================
# FINAL REPORT GENERATOR
# =========================================================
def generate_report(student, prediction):

    print("RECOMMENDATIONS")
    print("───────────────")

    recommendations = []

    # =====================================================
    # DYNAMIC RECOMMENDATIONS
    # =====================================================

    if student["stress_level"] >= 7:
        recommendations.append(
            "Practice stress management"
        )

    if student["sleep_hours"] <= 5:
        recommendations.append(
            "Improve sleep schedule"
        )

    if student["screen_time"] >= 8:
        recommendations.append(
            "Reduce screen time"
        )

    if student["social_support"] <= 4:
        recommendations.append(
            "Increase social interaction"
        )

    if student["study_hours_per_day"] >= 10:
        recommendations.append(
            "Reduce academic overload"
        )

    if student["physical_activity"] <= 2:
        recommendations.append(
            "Increase physical activity"
        )

    if prediction == "High":
        recommendations.append(
            "Seek counseling support if needed"
        )

    if not recommendations:
        recommendations.append(
            "Maintain current healthy lifestyle"
        )

    for tip in recommendations:
        print(f"✔ {tip}")

    

generate_report(
    user_student,
    user_pred
)

save_report(
    user_pred,
    user_conf
)

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
