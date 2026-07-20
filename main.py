from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from inference import MODEL_PATH, predict_new_data
import uvicorn

class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict churn probability for cleaned telco customer data.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "message": "Telco Customer Churn API",
        "status": "ok",
        "endpoints": ["/health", "/predict", "/docs"],
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not payload.records:
        raise HTTPException(status_code=400, detail="records list cannot be empty")

    try:
        raw_data = pd.DataFrame(payload.records)
        prediction_df = predict_new_data(raw_data, model_path=MODEL_PATH)
        return PredictionResponse(
            predictions=prediction_df.to_dict(orient="records")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
