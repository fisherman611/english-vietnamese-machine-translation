import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/filtered_cleaned_dataset.csv')
# data = pd.read_csv('data/cleaned_dataset.csv')

# First, split off 10% of the data (for train + test)
train, test = train_test_split(data, test_size=0.1, random_state=42)

# Second, split off 10% of the train data (for train + validation)
train, val = train_test_split(train, test_size=0.1, random_state=42)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Validation shape:", val.shape)

train.to_csv("data/train_filtered_cleaned_dataset.csv", index=False)
test.to_csv("data/test_filtered_cleaned_dataset.csv", index=False)
val.to_csv("data/val_filtered_cleaned_dataset.csv", index=False)

# train.to_csv("data/train_cleaned_dataset.csv", index=False)
# test.to_csv("data/test_cleaned_dataset.csv", index=False)
# val.to_csv("data/val_cleaned_dataset.csv", index=False)
