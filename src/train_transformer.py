import os
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

# Group by id_network_diagnostic (incident/case id)
if 'id_network_diagnostic' not in df.columns:
    raise ValueError("Expected 'id_network_diagnostic' column in dataset.")

# Set up column names
label_col = 'id_diagnostic_type'
group_col = 'id_network_diagnostic'

# Drop rows with missing values in key columns
df = df.dropna(subset=[group_col, 'description', label_col])

# Group by incident ID, keeping the description and label
grouped = df.groupby([group_col, label_col])['description'].first().reset_index()

label_encoder = LabelEncoder()
grouped['label'] = label_encoder.fit_transform(grouped[label_col])

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
    grouped[['description', 'label']],
    test_size=0.2,
    stratify=grouped['label'],
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
num_labels = grouped['label'].nunique()
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

trainer.train()

# 5. Save model, tokenizer, and label encoder
os.makedirs(MODEL_EXPORT_PATH, exist_ok=True)
model.save_pretrained(MODEL_EXPORT_PATH)
tokenizer.save_pretrained(TOKENIZER_EXPORT_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
print(f"Transformer model, tokenizer, and label encoder saved to {MODEL_EXPORT_PATH}, {TOKENIZER_EXPORT_PATH}, and {LABEL_ENCODER_PATH}")
