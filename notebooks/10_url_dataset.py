import pandas as pd

# Load phishing URL dataset
data = pd.read_csv(
    "datasets/PhiUSIIL_Phishing_URL_Dataset.csv"
)

print("First 5 records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum().sum())

print("\nLabel Distribution:")
print(data["label"].value_counts())

print("\nSample URLs:")
print(data["URL"].head())