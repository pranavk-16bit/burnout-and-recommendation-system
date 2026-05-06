import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

df = pd.read_csv(r"C:\Users\prana\Downloads\student_mental_health_burnout.csv")


df.columns = df.columns.str.strip().str.lower()

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== BURNOUT LEVEL COUNTS =====")
print(df["burnout_level"].value_counts())




df['burnout_level'] = df['burnout_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})


df = df.drop('student_id', axis=1)


df['stress_level'] = df['stress_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

df['sleep_quality'] = df['sleep_quality'].map({
    'Poor': 0,
    'Average': 1,
    'Good': 2
})

df['internet_quality'] = df['internet_quality'].map({
    'Poor': 0,
    'Average': 1,
    'Good': 2
})

df['year'] = df['year'].map({
    '1st': 1,
    '2nd': 2,
    '3rd': 3,
    '4th': 4
})

df = pd.get_dummies(
    df,
    columns=['gender', 'course'],
    drop_first=True
)

X = df.drop("burnout_level", axis=1)
y = df["burnout_level"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

print("\nAccuracy:",
      accuracy_score(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Train Accuracy:",
      rf_model.score(X_train, y_train))

print("Test Accuracy:",
      rf_model.score(X_test, y_test))

print("\nAccuracy:",
      accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

importance = rf_model.feature_importances_

feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_importance_df = feature_importance_df.sort_values(
    by='Importance',
    ascending=False
)
feature_importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\n==============================")
print("TOP FEATURE IMPORTANCE")
print("==============================")

print(feature_importance_df.head(10))

