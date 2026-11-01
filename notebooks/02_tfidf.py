import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Create TF-IDF object
vectorizer = TfidfVectorizer()

# Convert messages into numerical values
X = vectorizer.fit_transform(data["message"])

print("Original messages:")
print(data["message"].head())

print("\nTF-IDF Matrix Shape:")
print(X.shape)

print("\nFirst message as numbers:")
print(X[0].toarray())

print("\nWords learned by TF-IDF:")
print(vectorizer.get_feature_names_out()[:20])