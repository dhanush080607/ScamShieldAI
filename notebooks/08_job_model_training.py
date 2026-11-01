import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("datasets/fake_job_postings.csv")

# Select columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

data = data[text_columns + ["fraudulent"]].copy()

# Fill missing values
for column in text_columns:
    data[column] = data[column].fillna("")

# Combine text
data["job_text"] = data[text_columns].agg(" ".join, axis=1)

# Clean text
data["job_text"] = data["job_text"].str.lower().str.strip()

# Remove empty and duplicate postings
data = data[data["job_text"] != ""]
data = data.drop_duplicates(subset=["job_text"])
data = data.reset_index(drop=True)

# Input and output
X = data["job_text"]
y = data["fraudulent"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/fake_job_model.pkl"
)

print("\nFake Job Model Saved Successfully!")