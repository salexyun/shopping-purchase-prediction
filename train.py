"""Train and evaluate models on UCI online shopping sessions."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
SEED = 42
TARGET = "Revenue"
NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "SpecialDay",
]
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"
]
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = ["PageValues", "BounceRates", "ExitRates"]
SCORING = {"ap": "average_precision", "roc_auc": "roc_auc"}
HYPERPARAMETERS = {
    "Logistic regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    },
    "Random forest": {
        "model__max_depth": [6, 12, None],
        "model__min_samples_leaf": [5, 20],
    },
    "Histogram gradient boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [7, 15, 31],
    },
}


def load_data(path=DATA_PATH):
    frame = pd.read_csv(path)
    if not pd.api.types.is_bool_dtype(frame[TARGET]) or frame[TARGET].isna().any():
        raise ValueError("Revenue must contain a known TRUE/FALSE label for every session.")
    return frame


def split_data(frame):
    """Keep identical input profiles together so they cannot span holdouts or CV folds."""
    X = frame[FEATURES]
    y = frame[TARGET].astype(int)
    groups = pd.util.hash_pandas_object(X, index=False)
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer_cv.split(X, y, groups))
    return X.iloc[dev_idx], X.iloc[test_idx], y.iloc[dev_idx], y.iloc[test_idx], groups.iloc[dev_idx]


def build_pipeline(estimator, *, scale=False):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    features = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([("features", features), ("model", estimator)])


def tune_models(X, y, groups):
    """Compare models on the same development folds and refit the best on all development data."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    dummy = build_pipeline(DummyClassifier(strategy="prior"))
    baseline = cross_validate(dummy, X, y, groups=groups, cv=cv, scoring=SCORING)
    rows = [{
        "model": "Prior baseline",
        "cv_ap": float(baseline["test_ap"].mean()),
        "cv_ap_std": float(baseline["test_ap"].std()),
        "cv_roc_auc": float(baseline["test_roc_auc"].mean()),
        "params": {},
    }]
    estimators = {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=SEED),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=150, early_stopping=False, random_state=SEED
        ),
    }
    best_search = None
    best_name = ""
    for name, estimator in estimators.items():
        search = GridSearchCV(
            build_pipeline(estimator, scale=name == "Logistic regression"),
            HYPERPARAMETERS[name],
            scoring=SCORING,
            refit="ap",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X, y, groups=groups)
        index = search.best_index_
        rows.append({
            "model": name,
            "cv_ap": float(search.best_score_),
            "cv_ap_std": float(search.cv_results_["std_test_ap"][index]),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][index]),
            "params": search.best_params_,
        })
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name = name
            best_search = search

    return best_name, best_search.best_estimator_, sorted(rows, key=lambda row: -row["cv_ap"])


def score_holdout(model, X, y):
    scores = model.predict_proba(X)[:, 1]
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "positive_rate": float(y.mean()),
        "confusion_matrix_at_0_5": confusion_matrix(y, scores >= 0.5, labels=[0, 1]).tolist(),
        "n": len(y),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    X_dev, X_test, y_dev, y_test, groups_dev = split_data(load_data(args.data))
    name, model, comparison = tune_models(X_dev, y_dev, groups_dev)
    metrics = {
        "selected_model": name,
        "development_rows": len(y_dev),
        "cross_validation": comparison,
        "test": score_holdout(model, X_test, y_test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": name, "pipeline": model}, args.output, compress=3)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
