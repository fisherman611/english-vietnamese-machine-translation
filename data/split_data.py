import pandas as pd
from sklearn.model_selection import train_test_split

envi_data = pd.concat(
    [
        pd.read_csv("data/envi_dataset_part1.csv"),
        pd.read_csv("data/envi_dataset_part2.csv"),
        pd.read_csv("data/envi_dataset_part3.csv"),
    ]
)

# First, split off 1% of the data (for train + test)
train, test = train_test_split(envi_data, test_size=0.01, random_state=42)

# Second, split off 1% of the train data (for train + validation)
train, val = train_test_split(train, test_size=0.01, random_state=42)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Validation shape:", val.shape)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
val.to_csv("data/val.csv", index=False)
