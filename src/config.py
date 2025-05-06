import os

# Path to your CSV dataset
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset.csv')
# Path to export the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.joblib')
# Choose: 'random_forest' or 'logistic_regression'
MODEL_TYPE = 'random_forest'
