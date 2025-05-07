# Network Equipment Issue Classifier

This project classifies technical descriptions of network equipment issues into 18 diagnosis categories using a machine learning model.

## Project Structure

- `src/`: Source code (data loading, training, prediction)
- `data/`: Dataset CSV file
- `models/`: Exported trained model

## Quickstart

1. Place your dataset as `data/dataset.csv` (columns: `descripcion`, `diagnostico`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python src/train.py
   ```
4. Predict a new description:
   ```bash
   python src/predict.py "interface eth0 is down"
   ```
5. Run the API:
   ```bash
   python run.py
   ```
6. Make a POST request to `http://localhost:8000/predict` with a JSON body containing the description of the issue.
Invoke-WebRequest -Uri http://localhost:8000/predict `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body (Get-Content -Raw -Path ".\payload.json")

## Configuration

Edit `src/config.py` to change paths or model type.

## Extending
- Modular code for easy upgrades (e.g., try different models or preprocessors).

## Bonus: Advanced Models
- For better results, consider transformer models (e.g., BERT) using Hugging Face Transformers.
- Continual learning: retrain or fine-tune the model as new labeled data arrives.
