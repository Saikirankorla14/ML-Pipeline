# ML Pipeline API

A production-ready REST API for training and serving scikit-learn classification models, built with FastAPI.

## Project structure

```
ml_pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI routes
│   └── model.py       # Training, registry, inference
├── tests/
│   └── test_api.py    # Pytest suite
├── train.py           # Standalone training script
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Quickstart (local)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# API at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

## Quickstart (Docker + MLflow)

```bash
docker compose up --build
# API     → http://localhost:8000
# MLflow  → http://localhost:5000
```

## API reference

### Train a model

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "iris",
    "algorithm": "rf",
    "test_size": 0.2,
    "random_state": 42
  }'
```

**Datasets:** `iris` | `wine` | `breast_cancer`  
**Algorithms:** `rf` (Random Forest) | `lr` (Logistic Regression) | `svm` | `knn`

Response:
```json
{
  "model_id": "a1b2c3d4",
  "algorithm": "Random Forest",
  "dataset": "iris",
  "accuracy": 0.9667,
  "cv_mean": 0.9583,
  "cv_std": 0.0236,
  "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
  "classes": ["setosa", "versicolor", "virginica"]
}
```

### Run inference

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

Response:
```json
{
  "model_id": "a1b2c3d4",
  "prediction": "setosa",
  "confidence": 0.9812,
  "probabilities": {
    "setosa": 0.9812,
    "versicolor": 0.0143,
    "virginica": 0.0045
  }
}
```

Pass `"model_id": "a1b2c3d4"` to target a specific model; omit to use the latest.

### List models

```bash
curl http://localhost:8000/models
```

### Delete a model

```bash
curl -X DELETE http://localhost:8000/models/a1b2c3d4
```

## Standalone training script

```bash
python train.py --dataset wine --algorithm svm --test-size 0.25
```

With MLflow tracking (start `mlflow ui` first):
```bash
pip install mlflow
mlflow ui &
python train.py --dataset breast_cancer --algorithm rf --experiment my-experiment
```

## Run tests

```bash
pytest tests/ -v
```

## Extending the pipeline

**Add a new dataset**

In `app/model.py`, add to `DATASETS`:
```python
from sklearn.datasets import load_digits
DATASETS["digits"] = load_digits
```

**Add a new algorithm**

```python
from sklearn.tree import DecisionTreeClassifier
ALGORITHMS["dt"] = lambda: DecisionTreeClassifier(max_depth=5)
ALGO_NAMES["dt"] = "Decision Tree"
```

**Persist models to disk**

Replace the in-memory `_store` dict in `ModelRegistry` with joblib serialization:
```python
import joblib
joblib.dump(pipeline, f"models/{model_id}.pkl")
pipeline = joblib.load(f"models/{model_id}.pkl")
```
