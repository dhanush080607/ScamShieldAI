import pandas as pd

# Load dataset
data = pd.read_csv("datasets/fake_job_postings.csv")

# Select required columns
data = data[
    [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "fraudulent"
    ]
].copy()

# Fill missing text values
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    data[column] = data[column].fillna("")

# Combine all text columns
data["job_text"] = data[text_columns].agg(" ".join, axis=1)

# Convert text to lowercase
data["job_text"] = data["job_text"].str.lower()

# Remove unnecessary spaces
data["job_text"] = data["job_text"].str.strip()

# Remove empty job descriptions
data = data[data["job_text"] != ""]

# Remove duplicate job text
data = data.drop_duplicates(subset=["job_text"])

# Reset index
data = data.reset_index(drop=True)

# Display results
print("Cleaned Dataset:")
print(data[["job_text", "fraudulent"]].head())

print("\nFinal Dataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data["job_text"].isnull().sum())

print("\nJob Distribution:")
print(data["fraudulent"].value_counts())