import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from salary import api, model as m

JOBS = ["JANITOR", "JUNIOR", "SENIOR", "CEO"]


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "companyId": rng.choice(["COMP1", "COMP2"], n), "jobType": rng.choice(JOBS, n),
        "degree": rng.choice(["NONE", "MASTERS"], n), "major": rng.choice(["NONE", "MATH"], n),
        "industry": rng.choice(["OIL", "EDUCATION"], n),
        "yearsExperience": rng.integers(0, 25, n), "milesFromMetropolis": rng.integers(0, 100, n),
    })
    df["salary"] = 50 + 30 * df.jobType.map({j: i for i, j in enumerate(JOBS)}) + 2 * df.yearsExperience - 0.4 * df.milesFromMetropolis
    return df


def test_pipeline_learns_structure():
    df = synthetic()
    pipe = m.build_pipeline(max_iter=100).fit(df[m.FEATURES], df.salary)
    a, b = pipe.predict(pd.DataFrame([{**df.iloc[0][m.FEATURES].to_dict(), "jobType": j} for j in ("JANITOR", "CEO")]))
    assert b - a > 60


def test_load_training_data_drops_zero_salary(tmp_path):
    pd.DataFrame({"jobId": ["a", "b"], **{c: ["x", "x"] for c in m.CATEGORICAL}, "yearsExperience": [1, 2],
                  "milesFromMetropolis": [3, 4]}).to_csv(tmp_path / "train_features.csv", index=False)
    pd.DataFrame({"jobId": ["a", "b"], "salary": [0, 100]}).to_csv(tmp_path / "train_salaries.csv", index=False)
    assert m.load_training_data(tmp_path).salary.tolist() == [100]


@pytest.fixture
def client(monkeypatch):
    df = synthetic()
    pipe = m.build_pipeline(max_iter=50).fit(df[m.FEATURES], df.salary)
    meta = {"options": {c: sorted(df[c].unique()) for c in m.CATEGORICAL}, "insights": {}}
    monkeypatch.setattr(api, "artifacts", lambda: (pipe, {"mse": 1}, meta))
    return TestClient(api.app)


POST = {"companyId": "COMP1", "jobType": "CEO", "degree": "MASTERS", "major": "MATH", "industry": "OIL",
        "yearsExperience": 10, "milesFromMetropolis": 5}


def test_predict(client):
    r = client.post("/predict", json={"postings": [POST, {**POST, "jobType": "JANITOR"}]})
    assert r.status_code == 200
    ceo, jan = (p["salary_k"] for p in r.json()["predictions"])
    assert ceo > jan


def test_predict_rejects_unknown_category(client):
    assert client.post("/predict", json={"postings": [{**POST, "jobType": "ASTRONAUT"}]}).status_code == 422
