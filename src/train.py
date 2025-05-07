import os
import time
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
from sklearn.preprocessing import LabelEncoder
import joblib
from config import DATA_PATH, MODEL_PATH

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"  # Spanish BERT
MODEL_EXPORT_PATH = os.path.join(os.path.dirname(MODEL_PATH), "transformer_model")
TOKENIZER_EXPORT_PATH = os.path.join(os.path.dirname(MODEL_PATH), "transformer_tokenizer")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(MODEL_PATH), "label_encoder_transformer.joblib")

# 1. Load and preprocess multirow (step, result) format for transformer
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

# Get actual column names from the dataframe
label_col = df.columns[0]  # First column (id_diagnostic_type with potential BOM)
description_col = 'description'

# Drop rows with missing values in key columns
df = df.dropna(subset=[description_col, label_col])

# Create a clean dataframe with just the needed columns
df_clean = df[[description_col, label_col]].copy()

label_encoder = LabelEncoder()
df_clean['label'] = label_encoder.fit_transform(df_clean[label_col])

# 2. Tokenization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Tokenize the texts
    tokenized = tokenizer(
        examples['description'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors=None  # We want this to be a list of tensors
    )
    return tokenized

from sklearn.model_selection import train_test_split
from datasets import Dataset

train_df, test_df = train_test_split(
    df_clean,
    test_size=0.2,
    random_state=42
)
# Rename and fix dtype for HuggingFace compatibility
train_df = train_df.rename(columns={'label': 'labels'})
test_df = test_df.rename(columns={'label': 'labels'})
train_df['labels'] = train_df['labels'].astype(int)
test_df['labels'] = test_df['labels'].astype(int)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

dataset = {'train': train_dataset, 'test': test_dataset}

# 3. Model
num_labels = df_clean['label'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="single_label_classification"
)

# 4. Training
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=label_encoder.classes_, zero_division=0)
    print("\nClassification Report:\n", report)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start timing the training process
start_time = time.time()

# Get total row count
total_rows = len(dataset['train']) + len(dataset['test'])

trainer.train()

# Calculate total training time
total_time = time.time() - start_time
total_minutes = total_time / 60

# Print training statistics
print(f"\nTraining Statistics:")
print(f"Total training time: {total_minutes:.2f} minutes")
print(f"Total number of rows processed: {total_rows:,}")

# Save model, tokenizer and label encoder
os.makedirs(MODEL_EXPORT_PATH, exist_ok=True)
model.save_pretrained(MODEL_EXPORT_PATH)
tokenizer.save_pretrained(TOKENIZER_EXPORT_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
print(f"Transformer model, tokenizer, and label encoder saved to {MODEL_EXPORT_PATH}, {TOKENIZER_EXPORT_PATH}, and {LABEL_ENCODER_PATH}")
