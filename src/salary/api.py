import json
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import ARTIFACT_DIR, FEATURES

app = FastAPI(title="Salary Prediction API", version="1.0.0",
              description="Predict a job posting's salary (in $1000s) from role, education, industry, experience and location.")


class Posting(BaseModel):
    companyId: str = Field(..., examples=["COMP37"])
    jobType: str = Field(..., examples=["SENIOR"])
    degree: str = Field(..., examples=["MASTERS"])
    major: str = Field(..., examples=["ENGINEERING"])
    industry: str = Field(..., examples=["FINANCE"])
    yearsExperience: int = Field(..., ge=0, le=50, examples=[10])
    milesFromMetropolis: int = Field(..., ge=0, le=200, examples=[20])


class Batch(BaseModel):
    postings: list[Posting] = Field(..., min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    meta = json.loads((ARTIFACT_DIR / "insights.json").read_text())
    return model, metrics, meta


def _art():
    try:
        return artifacts()
    except FileNotFoundError:
        raise HTTPException(503, "model artifacts missing; run `python -m salary.train`")


@app.get("/health")
def health():
    _art()
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return _art()[1]


@app.get("/options")
def options():
    return _art()[2]["options"]


@app.get("/insights")
def insights():
    return _art()[2]["insights"]


@app.post("/predict")
def predict(batch: Batch):
    model, _, meta = _art()
    for i, p in enumerate(batch.postings):
        for col in ("jobType", "degree", "major", "industry", "companyId"):
            if getattr(p, col) not in meta["options"][col]:
                raise HTTPException(422, f"postings[{i}].{col}: unknown value {getattr(p, col)!r}")
    df = pd.DataFrame([p.model_dump() for p in batch.postings])[FEATURES]
    return {"predictions": [{"salary_k": round(float(v), 1)} for v in model.predict(df)]}
