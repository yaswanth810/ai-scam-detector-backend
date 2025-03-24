import pandas as pd

# Load dataset with correct encoding
df = pd.read_csv("scam_dataset.csv", encoding="ISO-8859-1")

# Drop extra columns
df = df.iloc[:, :2]  # Keep only first 2 columns

# Rename columns
df.columns = ["label", "text"]

# Convert labels to binary (ham → 0, spam → 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Save the cleaned dataset for training
df.to_csv("cleaned_dataset.csv", index=False)

print("\nDataset cleaned and saved as 'cleaned_dataset.csv'.")
