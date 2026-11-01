import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv(
    "datasets/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Clean messages
data["message"] = data["message"].str.lower().str.strip()

# Remove duplicates
data = data.drop_duplicates().reset_index(drop=True)

# Separate input and output
X = data["message"]
y = data["label"]

# Split dataset first
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test custom messages

print("\n--- ScamShield AI Message Tester ---")

while True:

    message = input("\nEnter a message (or type exit): ")

    if message.lower() == "exit":
        print("Thank you for using ScamShield AI!")
        break

    prediction = model.predict([message])[0]

    if prediction == 1:
        print("WARNING: This message looks like SPAM!")
    else:
        print("SAFE: This message looks NORMAL.")
