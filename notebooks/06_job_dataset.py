import pandas as pd

# Load fake job dataset
data = pd.read_csv("datasets/fake_job_postings.csv")

# Display first 5 rows
print("First 5 Job Postings:")
print(data.head())

# Dataset shape
print("\nDataset Shape:")
print(data.shape)

# Column names
print("\nColumns:")
print(data.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check genuine and fake jobs
print("\nJob Distribution:")
print(data["fraudulent"].value_counts())