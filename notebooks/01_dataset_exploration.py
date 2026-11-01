import pandas as pd

data = pd.read_csv(
    "datasets/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels into numbers
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Convert messages to lowercase
data["message"] = data["message"].str.lower()

# Remove unnecessary spaces
data["message"] = data["message"].str.strip()

# Check duplicate messages
print("\nDuplicate Messages:")
print(data.duplicated().sum())

print("\nUpdated Dataset:")
print(data.head())

print("\nUpdated Labels:")
print(data["label"].value_counts())

# Remove duplicate messages
data = data.drop_duplicates()

# Reset row indexes
data = data.reset_index(drop=True)

print("\nDataset after removing duplicates:")
print(data.shape)

print("\nDuplicate count after removal:")
print(data.duplicated().sum())