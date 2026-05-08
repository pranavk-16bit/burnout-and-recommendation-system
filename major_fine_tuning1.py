import os
import warnings
import joblib
import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# SETTINGS
# =========================================================

warnings.filterwarnings("ignore")

os.makedirs("visuals", exist_ok=True)

sns.set_theme(
    style="whitegrid",
    palette="flare",
    context="talk"
)

plt.rcParams["figure.figsize"] = (10, 6)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    r"C:\Users\prana\Downloads\student_mental_health_burnout.csv"
)

# =========================================================
# PREPROCESSING + FEATURE ENGINEERING
# =========================================================

maps = {
    "burnout_level": {"Low":0, "Medium":1, "High":2},
    "stress_level": {"Low":0, "Medium":1, "High":2},
    "sleep_quality": {"Poor":0, "Average":1, "Good":2},
    "internet_quality": {"Poor":0, "Average":1, "Good":2},
    "year": {"1st":1, "2nd":2, "3rd":3, "4th":4}
}

df.columns = df.columns.str.lower()

for col, mapping in maps.items():
    df[col] = df[col].map(mapping)

df.drop(
    ["student_id", "age"],
    axis=1,
    inplace=True
)

df = pd.get_dummies(
    df,
    columns=["gender", "course"],
    drop_first=True
)

df["mental_health_score"] = (
    df["anxiety_score"] +
    df["depression_score"] +
    df["academic_pressure_score"]
)

df["study_sleep_ratio"] = (
    df["daily_study_hours"] /
    (df["daily_sleep_hours"] + 1)
)

df["wellness_score"] = (
    df["physical_activity_hours"] +
    df["social_support_score"] -
    df["stress_level"]
)

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop("burnout_level", axis=1)
y = df["burnout_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
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

    "Logistic Regression":

        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
}

results = {}

for name, model in models.items():

    train_x = X_train_scaled if "Logistic" in name else X_train
    test_x = X_test_scaled if "Logistic" in name else X_test

    model.fit(train_x, y_train)

    pred = model.predict(test_x)

    results[name] = accuracy_score(y_test, pred)

    print(f"{name}: {results[name]:.4f}")

# =========================================================
# FINAL MODEL
# =========================================================

rf_model = models["Random Forest"]

rf_pred = rf_model.predict(X_test)

joblib.dump(
    rf_model,
    "burnout_model.pkl"
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    "Feature": X.columns,
    "Importance": rf_model.feature_importances_

}).sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(10)

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

# =========================================================
# STUDENT ANALYSIS
# =========================================================

sample = X_test.iloc[0]

labels = ["Low", "Medium", "High"]

prediction = labels[rf_pred[0]]

confidence = (
    rf_model.predict_proba(X_test)[0].max() * 100
)

risk_score = (

    sample["mental_health_score"] * 0.5 +

    sample["screen_time_hours"] +

    sample["stress_level"] * 2 -

    sample["daily_sleep_hours"]
)

severity = (

    "Low Risk" if risk_score < 8 else

    "Moderate Risk" if risk_score < 15 else

    "High Risk"
)

print("\n" + "="*40)
print("STUDENT BURNOUT ANALYSIS")
print("="*40)

print(f"Prediction : {prediction}")
print(f"Confidence : {confidence:.2f}%")
print(f"Risk Score : {risk_score:.2f}")
print(f"Severity   : {severity}")

# =========================================================
# SMART INSIGHTS
# =========================================================

checks = {

    "⚠ High Screen Time":
        sample["screen_time_hours"] > 8,

    "⚠ Poor Sleep":
        sample["daily_sleep_hours"] < 6,

    "⚠ High Stress":
        sample["stress_level"] == 2,

    "⚠ Low Wellness":
        sample["wellness_score"] < 2
}

print("\nRisk Factors")

for issue, condition in checks.items():

    if condition:
        print(issue)

# =========================================================
# RECOMMENDATIONS
# =========================================================

recommendations = {

    "✔ Reduce screen time":
        sample["screen_time_hours"] > 8,

    "✔ Improve sleep schedule":
        sample["daily_sleep_hours"] < 6,

    "✔ Practice stress management":
        sample["stress_level"] == 2,

    "✔ Increase physical activity":
        sample["wellness_score"] < 2
}

print("\nRecommendations")

for rec, condition in recommendations.items():

    if condition:
        print(rec)

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

sample_shap = X_test.sample(500, random_state=42)

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(sample_shap)

plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    sample_shap,
    plot_type="bar",
    show=False
)

plt.title(
    "SHAP Feature Importance",
    fontsize=18,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    "visuals/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# =========================================================
# REUSABLE SAVE FUNCTION
# =========================================================

def save_plot(title, file):

    plt.title(
        title,
        fontsize=18,
        weight="bold",
        pad=15
    )

    plt.tight_layout()

    plt.savefig(
    f"visuals/{file}.png",
    dpi=300,
    bbox_inches="tight"
)

    plt.show()
    plt.close()

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

plt.figure(figsize=(12,7))

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
    x="mental_health_score",
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

trend = (

    df.groupby("mental_health_score")

    ["screen_time_hours"]

    .mean()

    .rolling(5)

    .mean()

    .reset_index()
)

plt.figure(figsize=(10,6))

sns.lineplot(
    data=trend,
    x="mental_health_score",
    y="screen_time_hours",
    linewidth=4,
    color="#FF9800"
)

plt.fill_between(
    trend["mental_health_score"],
    trend["screen_time_hours"],
    alpha=0.2,
    color="#FF9800"
)

save_plot(
    "Mental Health vs Screen Time",
    "screen_time_trend"
)

# =========================================================
# CLEAN HEATMAP
# =========================================================

important_cols = [

    "mental_health_score",
    "study_sleep_ratio",
    "wellness_score",
    "screen_time_hours",
    "daily_sleep_hours",
    "stress_level",
    "burnout_level"
]

plt.figure(figsize=(10,7))

sns.heatmap(
    df[important_cols].corr(),
    annot=False,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

save_plot(
    "Key Feature Correlation",
    "correlation_heatmap"
)

# =========================================================
# BURNOUT SEVERITY PIE CHART
# =========================================================

severity_counts = pd.Series([

    "Low" if x < 8 else
    "Moderate" if x < 15 else
    "High"

    for x in df["mental_health_score"]

]).value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    severity_counts,
    labels=severity_counts.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("Set2")
)

save_plot(
    "Burnout Severity Distribution",
    "severity_distribution"
)

print("\nVisuals saved inside visuals/ folder")
print("Model saved as burnout_model.pkl")
