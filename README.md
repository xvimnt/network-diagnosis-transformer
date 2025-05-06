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

## Configuration

Edit `src/config.py` to change paths or model type.

## Extending
- Modular code for easy upgrades (e.g., try different models or preprocessors).

## Bonus: Advanced Models
- For better results, consider transformer models (e.g., BERT) using Hugging Face Transformers.
- Continual learning: retrain or fine-tune the model as new labeled data arrives.
