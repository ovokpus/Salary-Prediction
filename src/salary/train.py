"""Train the salary model and write artifacts (model, metrics, insights) to ARTIFACT_DIR."""
import argparse
import json
import time

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .model import ARTIFACT_DIR, CATEGORICAL, FEATURES, TARGET, build_pipeline, load_training_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None, help="Train on the first N rows (for quick runs).")
    args = ap.parse_args()

    t0 = time.time()
    df = load_training_data(nrows=args.nrows)
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.1, random_state=0)
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    baseline = X_te.join(y_te).groupby("jobType")[TARGET].transform("mean")  # notebook-style group-mean baseline
    metrics = {
        "model": "HistGradientBoostingRegressor",
        "n_train": len(X_tr), "n_holdout": len(X_te),
        "mse": round(mean_squared_error(y_te, pred), 2),
        "mae": round(mean_absolute_error(y_te, pred), 2),
        "r2": round(r2_score(y_te, pred), 4),
        "baseline_mse_jobtype_mean": round(mean_squared_error(y_te, baseline), 2),
        "train_seconds": round(time.time() - t0, 1),
    }
    insights = {
        c: df.groupby(c)[TARGET].mean().round(1).sort_values().to_dict()
        for c in ["jobType", "degree", "major", "industry"]
    }
    insights["yearsExperience"] = df.groupby("yearsExperience")[TARGET].mean().round(1).to_dict()
    options = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACT_DIR / "insights.json").write_text(json.dumps({"insights": insights, "options": options}, default=str))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
