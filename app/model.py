import uuid
import time
from typing import Optional

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


# ---------- Dataset loader ----------

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

ALGORITHMS = {
    "rf":  lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "lr":  lambda: LogisticRegression(max_iter=1000, random_state=42),
    "svm": lambda: SVC(probability=True, random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}

ALGO_NAMES = {
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "svm": "SVM",
    "knn": "K-Nearest Neighbors",
}


def load_dataset(name: str):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    data = DATASETS[name]()
    return data.data, data.target, list(data.feature_names), [str(c) for c in data.target_names]


# ---------- Registry ----------

class ModelRegistry:
    """In-memory model store. Swap with MLflow or a DB for production."""

    def __init__(self):
        self._store: dict[str, dict] = {}   # model_id -> record
        self._latest: Optional[str] = None

    # ---- train ----

    def train(self, dataset: str, algorithm: str, test_size: float, random_state: int) -> dict:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHMS)}")

        X, y, feature_names, classes = load_dataset(dataset)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ALGORITHMS[algorithm]()),
        ])

        # Cross-validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")

        # Fit on full training set, evaluate on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, y_pred))

        model_id = str(uuid.uuid4())[:8]
        record = {
            "model_id": model_id,
            "algorithm": ALGO_NAMES[algorithm],
            "algorithm_key": algorithm,
            "dataset": dataset,
            "accuracy": round(test_accuracy, 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "feature_names": feature_names,
            "classes": classes,
            "pipeline": pipeline,
            "trained_at": time.time(),
        }

        self._store[model_id] = record
        self._latest = model_id

        return {k: v for k, v in record.items() if k != "pipeline"}

    # ---- predict ----

    def predict(self, features: list[float], model_id: Optional[str] = None) -> dict:
        mid = model_id or self._latest
        if mid is None:
            raise ValueError("No trained model found. Call POST /train first.")
        if mid not in self._store:
            raise KeyError(f"Model '{mid}' not found in registry.")

        record = self._store[mid]
        pipeline = record["pipeline"]
        classes = record["classes"]

        X = np.array(features).reshape(1, -1)

        if X.shape[1] != len(record["feature_names"]):
            raise ValueError(
                f"Expected {len(record['feature_names'])} features, got {X.shape[1]}."
            )

        pred_idx = int(pipeline.predict(X)[0])
        proba = pipeline.predict_proba(X)[0]

        return {
            "model_id": mid,
            "prediction": classes[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, proba)},
        }

    # ---- list / delete ----

    def list_models(self) -> list[dict]:
        return [
            {k: v for k, v in rec.items() if k != "pipeline"}
            for rec in sorted(self._store.values(), key=lambda r: r["trained_at"], reverse=True)
        ]

    def delete(self, model_id: str) -> bool:
        if model_id not in self._store:
            return False
        del self._store[model_id]
        if self._latest == model_id:
            self._latest = next(iter(self._store), None)
        return True
