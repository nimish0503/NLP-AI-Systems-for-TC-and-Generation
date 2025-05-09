import pandas as pd

# Fix train.csv
train = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv")
train["Labels"] = train["Labels"].astype(int) - 1
train.to_csv("train.csv", index=False)
print("✅ Fixed train.csv")

# Fix val.csv
val = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/val.csv")
val["Labels"] = val["Labels"].astype(int) - 1
val.to_csv("val.csv", index=False)
print("✅ Fixed val.csv")

# Sanity check
print("\n📊 New val.csv label range:")
print("Min:", val["Labels"].min(), "| Max:", val["Labels"].max())
print(val["Labels"].value_counts())
