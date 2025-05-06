import sys
import joblib
from config import MODEL_PATH


def load_model(model_path):
    return joblib.load(model_path)


def predict_description(description, model_bundle):
    tfidf = model_bundle['tfidf']
    model = model_bundle['model']
    label_encoder = model_bundle['label_encoder']
    X = tfidf.transform([description])
    y_pred = model.predict(X)
    return label_encoder.inverse_transform(y_pred)[0]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'description text'")
        sys.exit(1)
    description = sys.argv[1]
    model_bundle = load_model(MODEL_PATH)
    prediction = predict_description(description, model_bundle)
    print(f"Predicted diagnosis: {prediction}")
