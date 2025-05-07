import sys
import os
import torch
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_PATH

MODEL_EXPORT_PATH = os.path.join(os.path.dirname(MODEL_PATH), "transformer_model")
TOKENIZER_EXPORT_PATH = os.path.join(os.path.dirname(MODEL_PATH), "transformer_tokenizer")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(MODEL_PATH), "label_encoder_transformer.joblib")

def load_all():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_EXPORT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_EXPORT_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return tokenizer, model, label_encoder

def predict(descriptions, tokenizer, model, label_encoder):
    inputs = tokenizer(descriptions, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return label_encoder.inverse_transform(preds)

if __name__ == "__main__":
    import json
    if len(sys.argv) < 2:
        print("Usage: python predict_transformer.py <input_json | json_string>")
        print("Input JSON must have a 'steps' key with a list of {step, result} objects.")
        sys.exit(1)
    input_arg = sys.argv[1]
    # Try to load as a file, else parse as string
    try:
        if os.path.exists(input_arg):
            with open(input_arg, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(input_arg)
    except Exception as e:
        print(f"Error parsing input: {e}")
        sys.exit(1)
    if 'steps' not in data or not isinstance(data['steps'], list):
        raise ValueError("Input JSON must have a 'steps' key with a list of {step, result} objects.")
    steps = data['steps']
    # Concatenate step/result pairs
    descripcion = ' | '.join(f"{s['step']}: {s['result']}" for s in steps if 'step' in s and 'result' in s)
    tokenizer, model, label_encoder = load_all()
    prediction = predict([descripcion], tokenizer, model, label_encoder)[0]
    print(f"Predicted id_diagnostic_type: {prediction}")
