import os, warnings, joblib, json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

from burnout_core.scoring import burnout_status
from burnout_core.security.consent import show_consent_screen
from burnout_core.security import storage as secure_storage
from burnout_core.features.engg_features import engineer_features

# =========================================================
# CONSENT CHECK
# =========================================================

if not show_consent_screen():
    print("\nYou declined data tracking. Exiting.")
    exit()

# =========================================================
# SETTINGS
# =========================================================

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")

MODEL_DIR = SCRIPT_DIR / "models"
REPORT_DIR = SCRIPT_DIR / "reports"
VISUAL_DIR = SCRIPT_DIR / "visuals"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)
CACHE_PATH = REPORT_DIR / "last_prediction_cache.json"
CACHE_MAX_AGE_MINUTES = 30


def load_cached_prediction():
    """Return the cached prediction if it's recent enough, else None."""

    if not CACHE_PATH.exists():
        return None

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    cached_time = datetime.strptime(cache["timestamp"], "%Y-%m-%d %H:%M:%S")
    age_minutes = (datetime.now() - cached_time).total_seconds() / 60

    if age_minutes > CACHE_MAX_AGE_MINUTES:
        return None

    return cache


def save_prediction_cache(student, pred, conf, probs, score, wellness):
    """Save the latest prediction so it can be reused if run again soon."""

    cache = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student": student,
        "user_pred": pred,
        "user_conf": float(conf),
        "probs": [float(p) for p in probs],
        "burnout_score": float(score),
        "wellness": float(wellness)
    }

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

        
sns.set_theme(style="whitegrid", palette="flare", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)

# =========================================================
# LOAD TRAINED MODEL (no training happens here)
# =========================================================

print("Loading trained model...")

best_model = joblib.load(MODEL_DIR / "burnout_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_columns = joblib.load(MODEL_DIR / "features.pkl")

with open(MODEL_DIR / "metadata.json") as f:
    metadata = json.load(f)

best_model_name = metadata["best_model_name"]
best_acc = metadata["best_acc"]
best_needs_scaling = metadata["needs_scaling"]

top_features = pd.read_csv(REPORT_DIR / "feature_importance.csv").head(10)
risk_correlation = pd.read_csv(REPORT_DIR / "risk_correlation.csv", index_col=0)["correlation"]

# =========================================================
# SAVE PLOT FUNCTION
# =========================================================

def save_plot(title, file):
    plt.title(title, fontsize=18, weight="bold")
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / f"{file}.png", dpi=300)
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
print("✔ Personalized Recommendations")
print("✔ Interactive Student Analysis")
print("✔ Report Generation")
print("✔ Visualization Dashboard")
print("\nInitializing system...\n")














# Clean training summary

# Model summary

print("MODEL LOADED")
print("────────────────────────")

print(
    f"Best Model : {best_model_name}"
)

print(
    f"Accuracy   : {best_acc:.2f}%\n"
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
        columns=feature_columns,
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
        REPORT_DIR / "student_report.txt",
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
        VISUAL_DIR / "burnout_trend.png",
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
        VISUAL_DIR / "sleep_burnout_trend.png",
        dpi=300
    )

    plt.show()
    plt.close()

def load_history():

    records = secure_storage.load_history()

    if not records:
        return pd.DataFrame(columns=[
            "date", "burnout_score", "risk_level", "confidence",
            "sleep_hours", "screen_time", "physical_activity",
            "wellness", "monitoring"
        ])

    history = pd.DataFrame(records)

    history["risk_level"] = (
        history["risk_level"]
        .fillna("Unknown")
    )

    return history


def early_warning():

    history = load_history()

    if len(history) < 7:
        return

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
        VISUAL_DIR / "burnout_dashboard.png",
        dpi=300
    )

    plt.show()

    plt.close()

def risk_factor_relationships():

    correlation = risk_correlation.sort_values()

    plt.figure(figsize=(10, 5))

    colors = ["green" if x < 0 else "crimson" for x in correlation.values]

    plt.barh(correlation.index, correlation.values, color=colors)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.xlabel("Correlation With Burnout Risk")

    plt.title("Risk Factor Relationships", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / "risk_factor_relationships.png", dpi=300)
    plt.show()
    plt.close()
    
# =========================================================
# INTERACTIVE PREDICTION
# =========================================================


cached = load_cached_prediction()

if cached:
    print(f"\nUsing cached prediction from {cached['timestamp']} (less than {CACHE_MAX_AGE_MINUTES} min old).")
    user_student = cached["student"]
    user_pred = cached["user_pred"]
    user_conf = cached["user_conf"]
    probs = np.array(cached["probs"])
    burnout_score = cached["burnout_score"]
else:
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

wellness = compute_wellness(user_student)

if not cached:
    save_prediction_cache(user_student, user_pred, user_conf, probs, burnout_score, wellness)
user_pred = apply_burnout_overrides(user_student, user_pred)




burnout_indicator(
    burnout_score
)
burnout_dashboard(
    user_student
)
risk_factor_relationships()

if not cached:
    secure_storage.save_history("unused", {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "burnout_score": float(burnout_score),
        "risk_level": user_pred,
        "confidence": float(user_conf),
        "sleep_hours": float(user_student["sleep_hours"]),
        "screen_time": float(user_student["screen_time"]),
        "physical_activity": float(user_student["physical_activity"]),
        "wellness": float(wellness),
        "monitoring": burnout_status(burnout_score)
    })
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
