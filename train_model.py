import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("reduced_datasetML.csv")

# Create binary target
df["Target_binary"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

# Encode categorical 'Course'
le = LabelEncoder()
df["Course_encoded"] = le.fit_transform(df["Course"])

# Select features
features = [
    "Total Grade", "Total Approved", "Tuition fees up to date",
    "Scholarship holder", "Debtor", "Age at enrollment",
    "Total Enrolled", "Total Evaluations", "Course_encoded"
]

X = df[features]
y = df["Target_binary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Calculate scale_pos_weight
neg, pos = np.bincount(y_train_bal)
scale_pos_weight = neg / pos

# Train model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train_bal, y_train_bal)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.4  # custom threshold to boost dropout recall
y_pred = (y_proba >= threshold).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Not Dropout", "Dropout"]))

# Save model and label encoder
joblib.dump(model, "student_dropout_xgb_final.pkl")
joblib.dump(le, "course_label_encoder.pkl")
print("✅ Model and encoder saved!")
