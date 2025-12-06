import pandas as pd

df = pd.read_csv('Dataset.csv')
split_idx = int(len(df) * 0.8)

train = df[:split_idx]
test = df[split_idx:]

train.to_csv('train_dataset.csv', index=False)
test.to_csv('test_dataset.csv', index=False)

print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")
