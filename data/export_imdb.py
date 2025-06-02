from datasets import load_dataset

ds = load_dataset("imdb")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("Đã lưu train.csv và test.csv")
