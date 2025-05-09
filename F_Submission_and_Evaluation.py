import pandas as pd
import json
from datasets import Dataset
from transformers import DistilBertTokenizerFast, Trainer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# Load tokenizer and fine-tuned model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./results")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["Interview Text"], padding="max_length", truncation=True)

# Load and preprocess validation data
val_df = pd.read_csv("data/val.csv")
#val_df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/val.csv")
val_df["Labels"] = val_df["Labels"].astype(int) - 1  # Shift labels from 1–8 → 0–7

# Tokenize val data
val_dataset = Dataset.from_pandas(val_df)
val_dataset = val_dataset.map(tokenize_function, batched=True)
val_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Predict on validation set
trainer = Trainer(model=model)
preds = trainer.predict(val_dataset)
label_ids = preds.predictions.argmax(-1)  # predicted class indices

# Save submission (with readable labels if needed)
label_decoder = {
    0: "pre_game_expectations",
    1: "post_game_reaction",
    2: "in_game_analysis",
    3: "career_reflection",
    4: "controversial_opinion",
    5: "injury_report",
    6: "training_insight",
    7: "off_field"
}
pred_labels_readable = [label_decoder[i] for i in label_ids]

submission = pd.DataFrame({
    "ID": val_df["ID"],
    "Labels": pred_labels_readable
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved from val.csv predictions.")

# ✅ Final corrected evaluation
y_true = val_df["Labels"].tolist()       # numeric true labels
y_pred = label_ids.tolist()              # numeric predictions

# Print debug info
print("\n🔍 Label Summary:")
print("True labels (unique):", set(y_true))
print("Predicted labels (unique):", set(y_pred))

print("\n🧪 First 10 Predictions vs True Labels:")
for i in range(min(10, len(y_true))):
    print(f"ID: {val_df['ID'].iloc[i]} | True: {y_true[i]} | Pred: {y_pred[i]}")

# Evaluate
if len(y_true) == len(y_pred) and len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    with open("results.json", "w") as f:
        json.dump({
            "f1_score": round(f1, 5),
            "accuracy": round(acc, 5)
        }, f)
    print(f"\n✅ results.json saved. Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
else:
    print("⚠️ Could not evaluate due to length mismatch.")
