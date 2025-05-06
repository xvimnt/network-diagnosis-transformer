import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from config import DATA_PATH, MODEL_PATH, MODEL_TYPE
from data_utils import load_data, preprocess_data


def train_and_evaluate():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, tfidf, label_encoder = preprocess_data(
        df, text_column='descripcion', label_column='diagnostico'
    )

    if MODEL_TYPE == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif MODEL_TYPE == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        'model': model,
        'tfidf': tfidf,
        'label_encoder': label_encoder
    }, MODEL_PATH)
    print(f"Model exported to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
