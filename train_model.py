"""Train the clinic regressor and export its fitted pipeline for browser inference."""

import csv
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = ["hour", "staff_on_shift", "patients_in_queue"]
DAY = "day_of_week"
TARGET = "wait_minutes"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def main():
    with (ROOT / "data/clinic-wait-times.csv").open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    features = np.array([[int(row[key]) for key in NUMERIC] + [row[DAY]] for row in rows], dtype=object)
    target = np.array([float(row[TARGET]) for row in rows])
    train_x, test_x, train_y, test_y, _, test_rows = train_test_split(
        features, target, np.arange(len(rows)), test_size=0.2, random_state=42
    )

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), [0, 1, 2]),
        ("weekday", OneHotEncoder(categories=[DAYS], sparse_output=False), [3]),
    ])
    network = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                           solver="lbfgs", max_iter=2000, tol=0.0001, max_fun=40000, random_state=42)
    pipeline = Pipeline([("preprocessing", preprocessing), ("network", network)])
    target_scaler = StandardScaler()
    scaled_train_y = target_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()
    pipeline.fit(train_x, scaled_train_y)
    scaled_predictions = pipeline.predict(test_x)
    predictions = np.maximum(0, target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel())
    median = float(np.median(train_y))
    baseline = np.full_like(test_y, median)
    errors = np.abs(predictions - test_y)
    worst = int(np.argmax(errors))
    matching_waits = target[np.all(features == test_x[worst], axis=1)]
    scaler = preprocessing.named_transformers_["numeric"]
    export = {
        "features": NUMERIC,
        "days": DAYS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "bounds": {key: [min(int(row[key]) for row in rows), max(int(row[key]) for row in rows)]
                   for key in NUMERIC},
        "targetMean": float(target_scaler.mean_[0]),
        "targetScale": float(target_scaler.scale_[0]),
        "layers": [{"weights": weights.tolist(), "bias": bias.tolist()}
                   for weights, bias in zip(network.coefs_, network.intercepts_)],
        "evaluation": {
            "trainCount": len(train_y), "testCount": len(test_y),
            "baselineMedian": median,
            "baselineMae": float(mean_absolute_error(test_y, baseline)),
            "modelMae": float(mean_absolute_error(test_y, predictions)),
            "mistake": {
                "inputs": {key: int(test_x[worst, index]) for index, key in enumerate(NUMERIC)}
                          | {DAY: str(test_x[worst, 3])},
                "actual": float(test_y[worst]), "prediction": float(predictions[worst]),
                "error": float(errors[worst]),
                "matchingWaits": sorted(matching_waits.tolist()),
            },
        },
        "verification": [
            {"inputs": {key: int(test_x[i, j]) for j, key in enumerate(NUMERIC)}
                       | {DAY: str(test_x[i, 3])},
             "prediction": float(predictions[i]), "csvLine": int(test_rows[i]) + 2}
            for i in range(len(test_y))
        ],
        "scikitLearnVersion": sklearn.__version__,
    }
    output = ROOT / "model/clinic-model.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(export, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(export["evaluation"], indent=2))


if __name__ == "__main__":
    main()


