from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.predict import load_all, predict
import uvicorn

app = FastAPI(title="Network Diagnosis Transformer API")

class Step(BaseModel):
    step: str
    result: str

class PredictionRequest(BaseModel):
    steps: List[Step]

class PredictionResponse(BaseModel):
    id_diagnostic_type: str

# Load model, tokenizer and label encoder at startup
tokenizer, model, label_encoder = load_all()

@app.post("/predict", response_model=PredictionResponse)
async def predict_diagnosis(request: PredictionRequest):
    try:
        # Concatenate step/result pairs
        descripcion = ' | '.join(f"{step.step}: {step.result}" for step in request.steps)
        
        # Make prediction
        prediction = predict([descripcion], tokenizer, model, label_encoder)[0]
        
        return PredictionResponse(id_diagnostic_type=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
