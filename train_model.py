"""Train and evaluate the sales demand estimation pipeline."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sales_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "sales_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
EVALUATION_FILE = MODEL_DIR / "evaluation_predictions.csv"
FEATURES = ["product_category", "unit_price", "discount_pct", "marketing_spend", "day_of_week", "month"]
NUMERIC_FEATURES = ["unit_price", "discount_pct", "marketing_spend", "day_of_week", "month"]
CATEGORICAL_FEATURES = ["product_category"]


def train() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}. Run: python generate_data.py")

    data = pd.read_csv(DATA_FILE)
    required = FEATURES + ["units_sold", "order_date"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data = data.dropna(subset=required).sort_values("order_date").reset_index(drop=True)
    if len(data) < 30:
        raise ValueError("At least 30 valid rows are required to train the model.")

    # Keep the most recent 20% as the holdout set to approximate future forecasting.
    split_index = max(1, min(len(data) - 1, int(len(data) * 0.8)))
    train_df = data.iloc[:split_index]
    test_df = data.iloc[split_index:]
    X_train, y_train = train_df[FEATURES], train_df["units_sold"].astype(float)
    X_test, y_test = test_df[FEATURES], test_df["units_sold"].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    predictions = np.maximum(0, pipeline.predict(X_test))
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    metrics = {
        "model_name": "RandomForestRegressor",
        "target": "units_sold",
        "rows_total": int(len(data)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "split_method": "chronological 80/20 holdout",
        "data_type": "synthetic sample data",
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    evaluation = pd.DataFrame({
        "order_date": test_df["order_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "product_category": test_df["product_category"].to_numpy(),
        "actual_units": y_test.to_numpy(dtype=float),
        "predicted_units": np.round(predictions, 2),
    })
    evaluation.to_csv(EVALUATION_FILE, index=False)
    print("Model training complete")
    print(f"Model: {metrics['model_name']}")
    print(f"Rows: {metrics['rows_total']} total | {metrics['train_rows']} train | {metrics['test_rows']} test")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"R²: {metrics['r2']}")
    print(f"Saved model: {MODEL_FILE}")
    return metrics


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    train()
