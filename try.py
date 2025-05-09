# Section B: Model 2 - DistilBERT with Hugging Face

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Loading data for preprocessing
df = pd.read_csv("/Users/nimishmathur/Desktop/NLP/Final Exam/data/train.csv")

# Ensuring labels are integers
df['Labels'] = df['Labels'].astype(int)

# Split the dataset - 90% train, 10% test
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['Labels'], random_state=42)

# 2. Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Tokenization with DistilBERT tokenizer

# Load a pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function to encode text into model-ready input
def tokenize_function(example):
    return tokenizer(example["Interview Text"], padding="max_length", truncation=True)

# Apply tokenization to the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 4. Preprare for Model Training

# Rename target column to 'labels' for compatibility with Hugging Face Trainer
train_dataset = train_dataset.rename_column("Labels", "labels")
test_dataset = test_dataset.rename_column("Labels", "labels")

# Set format to PyTorch tensors
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Load pre trained DistilBERT Model

# Define Number of classes and load model for classification
num_labels = df["Labels"].nunique()
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# 6. Define Evaluation Metrics

# Compute Accuracy score and F1 score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# 7. Define Training Arguments
# Configure Basic Training Parameters
training_args = TrainingArguments(
    output_dir="./results", # Save model logs here
    num_train_epochs=3, # Number of epochs
    per_device_train_batch_size=8, # Training batch size
    per_device_eval_batch_size=16, # Evalution batch size
    logging_dir="./logs" # Log Directory
)

# 8. Initialize HuggingFace Trainer

# Set up the Trainer with model, training arguments, datasets, and evaluation metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9. Train and evaluate the model

trainer.train() # Start training

# Evaluate the model on the test set
metrics = trainer.evaluate()
print(metrics)

# 10. Save the model and tokenizer

trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
