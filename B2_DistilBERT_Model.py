# B2_DistilBERT_Model_Weighted_Optimized.py: DistilBERT with class weights, tuned LR, and oversampling

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Load and label-shift dataset
df = pd.read_csv("/data/train.csv")
#df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv")
df["Labels"] = df["Labels"].astype(int) - 1

# ✅ Oversample minority classes
def oversample_dataframe(df):
    max_count = df["Labels"].value_counts().max()
    balanced_df = pd.concat([
        resample(df[df["Labels"] == label], replace=True, n_samples=max_count, random_state=42)
        for label in df["Labels"].unique()
    ])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_df = oversample_dataframe(df)
train_df, val_df = train_test_split(balanced_df, stratify=balanced_df["Labels"], test_size=0.1, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["Labels"]), y=train_df["Labels"])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer setup
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["Interview Text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("Labels", "labels")
val_dataset = val_dataset.rename_column("Labels", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Model setup
num_labels = df["Labels"].nunique()
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=3e-5
)

# Initialize and run trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

trainer.save_model("./results")
tokenizer.save_pretrained("./results")