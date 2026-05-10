from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os

from app.model import ModelRegistry

app = FastAPI(
    title="ML Pipeline API",
    description="Train and serve scikit-learn models via REST",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()

# Serve frontend (if frontend/ folder exists next to app/)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ---------- Schemas ----------

class TrainRequest(BaseModel):
    dataset: str = Field("iris", description="iris | wine | breast_cancer")
    algorithm: str = Field("rf", description="rf | lr | svm | knn")
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    random_state: int = 42


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector matching the trained dataset")
    model_id: Optional[str] = Field(None, description="Specific model ID; uses latest if omitted")


class TrainResponse(BaseModel):
    model_id: str
    algorithm: str
    dataset: str
    accuracy: float
    cv_mean: float
    cv_std: float
    feature_names: list[str]
    classes: list[str]


class PredictResponse(BaseModel):
    model_id: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]


# ---------- Routes ----------

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "ui": "/ui"}


@app.get("/models")
def list_models():
    """List all trained models in the registry."""
    return registry.list_models()


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """Train a model and register it."""
    try:
        result = registry.train(
            dataset=req.dataset,
            algorithm=req.algorithm,
            test_size=req.test_size,
            random_state=req.random_state,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run inference against a trained model."""
    try:
        result = registry.predict(
            features=req.features,
            model_id=req.model_id,
        )
        return result
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    """Remove a model from the registry."""
    removed = registry.delete(model_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"deleted": model_id}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)