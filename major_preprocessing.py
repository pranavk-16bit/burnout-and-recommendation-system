import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\prana\Downloads\student_mental_health_burnout.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# 🔥 Convert target variable (burnout_level → numeric)
df['burnout_level'] = df['burnout_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

# Drop useless column
df = df.drop('student_id', axis=1)

# 🔥 One-hot encoding for categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop("burnout_level", axis=1)
y = df["burnout_level"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))



