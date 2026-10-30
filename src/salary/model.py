"""Model definition and data loading, distilled from 'Salary Prediction Project.ipynb'."""
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", Path(__file__).resolve().parents[2] / "artifacts"))

CATEGORICAL = ["companyId", "jobType", "degree", "major", "industry"]
NUMERICAL = ["yearsExperience", "milesFromMetropolis"]
FEATURES = CATEGORICAL + NUMERICAL
TARGET = "salary"


def load_training_data(data_dir: Path = DATA_DIR, nrows: int | None = None) -> pd.DataFrame:
    X = pd.read_csv(data_dir / "train_features.csv", nrows=nrows)
    y = pd.read_csv(data_dir / "train_salaries.csv", nrows=nrows)
    df = X.merge(y, on="jobId")
    # The notebook's EDA found a handful of zero salaries; they are invalid and dropped.
    return df[df[TARGET] > 0].reset_index(drop=True)


def build_pipeline(max_iter: int = 300) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63,
        categorical_features=list(range(len(CATEGORICAL))), random_state=0,
    )
    return Pipeline([("pre", pre), ("gbm", model)])
