import os, warnings, joblib,numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from burnout_core.scoring import burnout_status
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from datetime import datetime
from sklearn.pipeline import make_pipeline


# =========================================================
# SETTINGS
# =========================================================

warnings.filterwarnings("ignore")
from pathlib import Path
for folder in map(Path, [
    "reports",
    "models",
    "visuals"
]):
    folder.mkdir(exist_ok=True)

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
# LOAD DATA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

BURNOUT_DATA_PATH = os.getenv("BURNOUT_DATA_PATH")

df = pd.read_csv(BURNOUT_DATA_PATH)
df = df.sample(n=50000, random_state=42)

# =========================================================
# SYSTEM BANNER
# =========================================================

print("\n" + "=" * 60)
print(" AI-POWERED STUDENT BURNOUT DETECTION SYSTEM ")
print("=" * 60)
print("\nFeatures Included:")
print("✔ Burnout Prediction")
print("✔ Personalized Recommendations")
print("✔ Interactive Student Analysis")
print("✔ Report Generation")
print("✔ Visualization Dashboard")
print("\nInitializing system...\n")



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
def engineer_features(d):
    d["stress_sleep_ratio"] = d["stress_level"] / (d["sleep_hours"] + 1)
    d["mental_pressure"] = d["anxiety_score"] + d["depression_score"] + d["exam_pressure"]
    d["wellness_score"] = d["physical_activity"] + d["social_support"] - d["stress_level"]
    d["digital_overload"] = d["screen_time"] * d["internet_usage"]
    d["sleep_quality"] = d["sleep_hours"] / (d["screen_time"] + 1)
    d["stress_index"] = d["stress_level"] * d["exam_pressure"]
    d["lifestyle_balance"] = d["physical_activity"] + d["social_support"] - d["screen_time"]
    return d

df = engineer_features(df)   # <-- this line was missing

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
    HistGradientBoostingClassifier(
        max_iter=40,
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
files = {
    "burnout_model.pkl": best_model,
    "scaler.pkl": scaler,
    "features.pkl": X.columns.tolist()
}

for name, obj in files.items():
    joblib.dump(obj, f"models/{name}")

# CROSS VALIDATION
best_needs_scaling = models[best_model_name][1]
cv_pipeline = make_pipeline(StandardScaler(), best_model) if best_needs_scaling else best_model
scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5)




best_acc = results[
    best_model_name
] * 100

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


f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

# =========================================================
# FEATURE IMPORTANCE DATA
# =========================================================

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_).mean(axis=0)
else:
    importances = np.zeros(len(X.columns))
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
top_features = importance_df.head(10)

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
    )




# =========================================================
# REUSABLE PREDICTION FUNCTION
# =========================================================

RISK_LABELS = {

0:"Low",

1:"Medium",

2:"High"
}

def predict_student(student_data):

    input_df = pd.DataFrame([student_data])

    # Add missing columns automatically
    input_df = input_df.reindex(
        columns=X.columns,
        fill_value=0
)
    

    prediction = best_model.predict(input_df)[0]

    probs = best_model.predict_proba(input_df)[0]

    return (
        RISK_LABELS[prediction],
        probs.max() * 100,
        probs
)

# =========================================================
# RISK DISPLAY SYSTEM
# =========================================================
RISK_EMOJIS = {

"Low":"🟢 LOW RISK",

"Medium":"🟠 MODERATE RISK",

"High":"🔴 HIGH RISK"

}
def risk_emoji(level):
    return RISK_EMOJIS[level]

def confidence_bar(conf):

    filled = round(conf / 10)

    return (
        "█" * filled +
        "░" * (10 - filled)
    )
    

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

def save_history(score,prediction,confidence,student,wellness,monitoring):
    history_file = (
        "reports/student_history.csv"
    )

    new_data = pd.DataFrame({

    "date":[
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ],

    "burnout_score":[score],

    "risk_level":[prediction],

    "confidence":[confidence],

    "sleep_hours":[
        student["sleep_hours"]
    ],

    "screen_time":[
        student["screen_time"]
    ],

    "physical_activity":[
        student["physical_activity"]
    ],

    "wellness":[wellness],

    "monitoring":[monitoring]

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
    updated.to_excel(
    "reports/student_history.xlsx",
    index=False
)

def burnout_trend():

    history = load_history()

    history["date"] = pd.to_datetime(
        history["date"],
        format="mixed"
    )

    history["date_label"] = (
        history["date"]
        .dt.strftime("%d-%b")
    )

    history = history.tail(7)

    if len(history) < 2:
        return

    plt.figure(figsize=(14,6))

    plt.plot(
    history["date_label"],
    history["burnout_score"],
        marker="o",
        linewidth=4,
        markersize=10,
        color="darkorange"
    )

    plt.fill_between(
    history["date_label"],
    history["burnout_score"],
        alpha=0.25,
        color="orange"
    )

    plt.grid(alpha=0.3)

    plt.ylabel("Burnout Score")
    plt.xlabel("Date")
    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "visuals/burnout_trend.png",
        dpi=300
    )

    plt.show()
    plt.close()


def sleep_burnout_trend():

    history = load_history()

    history["date"] = pd.to_datetime(
        history["date"],
        format="mixed"
    )

    history["date_label"] = (
        history["date"]
        .dt.strftime("%d-%b")
    )

    history = history.tail(7)

    if len(history) < 2:
        return

    fig, ax1 = plt.subplots(
        figsize=(14,6)
    )

    ax1.plot(
    history["date_label"],
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
    history["date_label"],
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

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "visuals/sleep_burnout_trend.png",
        dpi=300
    )

    plt.show()
    plt.close()

def load_history():

    history = pd.read_csv(
        "reports/student_history.csv"
    )

    history["risk_level"] = (
        history["risk_level"]
        .fillna("Unknown")
    )

    return history


def early_warning():

    history_file = (
        "reports/student_history.csv"
    )

    if not os.path.exists(history_file):
        return

    history = load_history()

    if len(history) < 7:
        return

    history = load_history()
    current_score = history[
        "burnout_score"
    ].iloc[-1]

    weekly_average = history[
        "burnout_score"
    ].tail(7).mean()

    increase = (
        (current_score - weekly_average)
        /
        weekly_average
    ) * 100

    if current_score > weekly_average:

        print("\n⚠ EARLY WARNING SYSTEM")
        print("─────────────────────")

        print(
            "Burnout Risk Increasing"
        )

        print(
            f"Trend has increased "
            f"{increase:.1f}% "
            f"during the last 7 records."
        )

        print(
            "Monitor sleep, stress and workload."
        )

    else:

        print("\n✅ EARLY WARNING SYSTEM")
        print("─────────────────────")

        print(
            "No rising burnout trend detected."
        )


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

def get_student_input():
    print("\n" + "=" * 40)
    print("ENTER STUDENT DETAILS")
    print("=" * 40)

    while True:
        study = ask("Study Hours Per Day (0-16): ", 0, 16)
        screen = ask("Screen Time Hours (0-16): ", 0, 16)
        sleep = ask("Sleep Hours (0-12): ", 0, 12)

        if study + screen + sleep <= 24:
            break
        print(f"❌ Study + Screen + Sleep = {study+screen+sleep} hours, "
              f"which exceeds 24 in a day. Please re-enter.")

    mental_health = ask("Mental Health Score (1-10): ", 1, 10, int)
    stress = max(1, 11 - mental_health)

    anxiety = max(1, round((11 - mental_health) * 0.9))
    depression = max(1, round((11 - mental_health) * 0.8))
    support = mental_health

    activity = ask("Physical Activity Hours (0-6): ", 0, 6)

    # NEW: ask these directly instead of copying other values
    exam_pressure = ask("Exam Pressure Level (1-10): ", 1, 10, int)
    internet_usage = ask("Internet Usage Hours Per Day (0-16): ", 0, 16)

    gender = ask_gender()

    student = {
        "age": 21,
        "academic_year": 3,
        "study_hours_per_day": study,
        "screen_time": screen,
        "sleep_hours": sleep,
        "stress_level": stress,
        "anxiety_score": anxiety,
        "depression_score": depression,
        "social_support": support,
        "exam_pressure": exam_pressure,        # real value now
        "internet_usage": internet_usage,      # real value now
        "physical_activity": activity,
        "mental_health_score": mental_health,
        "financial_stress": 5,
        "family_expectation": 5,
        "academic_performance": 7,
        "gender_Male": 1 if gender == "male" else 0,
    }
    return engineer_features(student)



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
def apply_burnout_overrides(student, prediction):
    if student["sleep_hours"] < 4 or student["screen_time"] > 12 or student["study_hours_per_day"] > 12:
        return "High"
    return prediction

user_pred = apply_burnout_overrides(user_student, user_pred)

def compute_wellness(student):
    sleep_component = min(student["sleep_hours"], 9) / 9        # 9hrs = ideal, capped
    activity_component = min(student["physical_activity"], 5) / 5
    support_component = student["social_support"] / 10
    screen_penalty = min(student["screen_time"], 12) / 12        # more screen = worse

    wellness = (
        sleep_component * 4 +
        activity_component * 3 +
        support_component * 3 -
        screen_penalty * 3
    )
    return max(0, min(wellness, 10))  # clamp to 0-10

wellness = compute_wellness(user_student)   # <-- this line replaces your old inline formula

user_pred = apply_burnout_overrides(user_student, user_pred)




burnout_indicator(
    burnout_score
)
burnout_dashboard(
    user_student
)
risk_factor_relationships()

save_history(

    burnout_score,

    user_pred,

    user_conf,

    user_student,

    wellness,

    burnout_status(burnout_score)

)
burnout_trend()

sleep_burnout_trend()


early_warning()


def health_bar(score, maximum):
    score = max(0, min(score, maximum))  # clamp before rendering
    filled = round((score / maximum) * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:.1f}/{maximum}"

def health_grade(score):

    if score >= 8:
        return "A"

    if score >= 6:
        return "B"

    if score >= 4:
        return "C"

    return "D"

stress_rating = 11 - user_student["stress_level"]

sleep_rating = min(5, round(user_student["sleep_hours"] / 2))

mental_rating = round(user_student["mental_health_score"] / 2)

activity_rating = min(5, round(user_student["physical_activity"]))

screen_rating = max(1, 6 - round(user_student["screen_time"]))

print("\n" + "=" * 42)
print("      STUDENT HEALTH SCORECARD")
print("=" * 42)

print(f"""
Burnout Risk        {risk_emoji(user_pred)}

Overall Wellness    {wellness:.1f}/10

Sleep
    {health_bar(user_student["sleep_hours"],10)}

Stress
    {health_bar(11-user_student["stress_level"],10)}

Mental Health
    {health_bar(user_student["mental_health_score"],10)}

Activity
    {health_bar(user_student["physical_activity"],10)}

Screen Time
    {health_bar(10-user_student["screen_time"],10)}

Confidence     : {confidence_bar(user_conf)}

                  {user_conf:.2f}%

Overall Health Grade

        {health_grade(wellness)}
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

print("\nBURNOUT RISK BREAKDOWN")
print("══════════════════════")

top3 = top_features.head(3).copy()

top3["Percent"] = (
    top3["Importance"]
    /
    top3["Importance"].sum()
) * 100

for row in top3.itertuples():

    name = feature_names.get(
        row.Feature,
        row.Feature
    )

    blocks = "█" * round(row.Percent / 10)

    print(
        f"{name:<25}"
        f"{blocks:<10}"
        f"{row.Percent:.1f}%"
    )


history = load_history()

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


print(
    history[
        ["date", "burnout_score", "risk_level"]
    ].tail(5)
)

# =========================================================
# FINAL REPORT GENERATOR
# =========================================================
def generate_report(student, prediction):

    print("\nRECOMMENDATIONS")
    print("════════════════════════════")

    recommendations = []

    # -------------------------
    # Sleep
    # -------------------------
    sleep = student["sleep_hours"]

    if sleep < 6:
        recommendations.append("🔴 Increase sleep by 2 hours daily.")
    elif sleep < 7:
        recommendations.append("🟠 Target 7-8 hours of sleep.")
    else:
        recommendations.append("🟢 Excellent sleeping habit.")

    # -------------------------
    # Stress
    # -------------------------
    stress = student["stress_level"]

    if stress >= 8:
        recommendations.append("🔴 Practice stress management immediately.")
    elif stress >= 5:
        recommendations.append("🟠 Include 15 minutes of meditation.")
    else:
        recommendations.append("🟢 Stress level is under control.")

    # -------------------------
    # Screen Time
    # -------------------------
    screen = student["screen_time"]

    if screen > 8:
        recommendations.append("🔴 Reduce screen time by at least 2 hours.")
    elif screen > 5:
        recommendations.append("🟠 Take a 10-minute break every hour.")
    else:
        recommendations.append("🟢 Healthy screen usage.")

    # -------------------------
    # Physical Activity
    # -------------------------
    activity = student["physical_activity"]

    if activity < 2:
        recommendations.append("🔴 Exercise at least 30 minutes daily.")
    elif activity < 5:
        recommendations.append("🟠 Increase physical activity.")
    else:
        recommendations.append("🟢 Excellent activity level.")

    # -------------------------
    # Social Support
    # -------------------------
    support = student["social_support"]

    if support < 4:
        recommendations.append("🟠 Spend more time with friends and family.")
    else:
        recommendations.append("🟢 Maintain your social support.")

    # -------------------------
    # Study Hours
    # -------------------------
    study = student["study_hours_per_day"]

    if study > 10:
        recommendations.append("🟠 Reduce academic workload.")
    elif study < 3:
        recommendations.append("🟢 Maintain consistent study habits.")

    # -------------------------
    # Extra Advice
    # -------------------------
    recommendations.append("💧 Drink 2-3 liters of water daily.")
    recommendations.append("🥗 Maintain a balanced diet.")
    recommendations.append("🚶 Walk outdoors for 20 minutes.")
    recommendations.append("🧘 Stretch every hour.")
    recommendations.append("😴 Avoid mobile phone before bedtime.")

    # -------------------------
    # Print Recommendations
    # -------------------------
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
