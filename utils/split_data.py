import pandas as pd
from sklearn.model_selection import train_test_split

envi_data = pd.read_csv('data/dataset_300k.csv')
# envi_data = pd.read_csv('data/dataset_500k.csv')

# First, split off 10% of the data (for train + test)
train, test = train_test_split(envi_data, test_size=0.1, random_state=42)

# Second, split off 10% of the train data (for train + validation)
train, val = train_test_split(train, test_size=0.1, random_state=42)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Validation shape:", val.shape)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
val.to_csv("data/val.csv", index=False)
