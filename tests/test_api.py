import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------- helpers ----------

def train_model(dataset="iris", algorithm="rf"):
    res = client.post("/train", json={
        "dataset": dataset,
        "algorithm": algorithm,
        "test_size": 0.2,
        "random_state": 42,
    })
    assert res.status_code == 200, res.text
    return res.json()


IRIS_SAMPLE = [5.1, 3.5, 1.4, 0.2]
WINE_SAMPLE  = [13.2, 1.78, 2.14, 11.2, 0.28, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
CANCER_SAMPLE = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
                 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601,
                 0.1189]


# ---------- root ----------

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ---------- train ----------

@pytest.mark.parametrize("dataset,algo,n_features", [
    ("iris", "rf", 4),
    ("iris", "lr", 4),
    ("iris", "svm", 4),
    ("iris", "knn", 4),
    ("wine", "rf", 13),
    ("breast_cancer", "rf", 30),
])
def test_train(dataset, algo, n_features):
    res = train_model(dataset=dataset, algorithm=algo)
    assert 0.7 < res["accuracy"] <= 1.0
    assert len(res["feature_names"]) == n_features
    assert len(res["classes"]) >= 2


def test_train_bad_dataset():
    res = client.post("/train", json={"dataset": "bogus", "algorithm": "rf"})
    assert res.status_code == 400


def test_train_bad_algorithm():
    res = client.post("/train", json={"dataset": "iris", "algorithm": "bogus"})
    assert res.status_code == 400


# ---------- predict ----------

def test_predict_uses_latest():
    train_model("iris", "rf")
    res = client.post("/predict", json={"features": IRIS_SAMPLE})
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] in ["setosa", "versicolor", "virginica"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_explicit_model_id():
    trained = train_model("iris", "rf")
    mid = trained["model_id"]
    res = client.post("/predict", json={"features": IRIS_SAMPLE, "model_id": mid})
    assert res.status_code == 200
    assert res.json()["model_id"] == mid


def test_predict_wrong_feature_count():
    train_model("iris", "rf")
    res = client.post("/predict", json={"features": [1.0, 2.0]})  # iris needs 4
    assert res.status_code == 400


def test_predict_missing_model():
    res = client.post("/predict", json={"features": IRIS_SAMPLE, "model_id": "nonexistent"})
    assert res.status_code == 400


# ---------- list / delete ----------

def test_list_models():
    train_model()
    res = client.get("/models")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
    assert len(res.json()) >= 1


def test_delete_model():
    trained = train_model()
    mid = trained["model_id"]
    res = client.delete(f"/models/{mid}")
    assert res.status_code == 200

    res = client.post("/predict", json={"features": IRIS_SAMPLE, "model_id": mid})
    assert res.status_code == 400


def test_delete_nonexistent():
    res = client.delete("/models/doesnotexist")
    assert res.status_code == 404
