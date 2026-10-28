"""Flask web application for AI-Based Sales Prediction and Business Analytics."""
from __future__ import annotations

from datetime import date, datetime
import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sales_data.csv"
MODEL_FILE = BASE_DIR / "models" / "sales_model.joblib"
METRICS_FILE = BASE_DIR / "models" / "metrics.json"
EVALUATION_FILE = BASE_DIR / "models" / "evaluation_predictions.csv"
DB_FILE = BASE_DIR / "database" / "sales.db"
CATEGORIES = ["Electronics", "Clothing", "Grocery", "Home & Kitchen", "Beauty & Personal Care"]

app = Flask(__name__)
app.secret_key = "local-sales-prediction-demo-change-before-deployment"


@contextmanager
def get_db():
    """Yield a configured SQLite connection and always close it."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_timestamp TEXT NOT NULL,
                selected_date TEXT NOT NULL,
                product_category TEXT NOT NULL,
                unit_price REAL NOT NULL,
                discount_pct REAL NOT NULL,
                marketing_spend REAL NOT NULL,
                predicted_units INTEGER NOT NULL,
                estimated_revenue REAL NOT NULL
            )
        """)
        conn.commit()


def load_metrics() -> dict[str, Any]:
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_sales_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Sales sample dataset is missing. Run `python generate_data.py` first.")
    data = pd.read_csv(DATA_FILE)
    if "order_date" in data.columns:
        data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce").fillna(0)
    data["units_sold"] = pd.to_numeric(data["units_sold"], errors="coerce").fillna(0)
    return data.dropna(subset=["order_date"])


def load_model():
    if not MODEL_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_FILE)
    except (OSError, ValueError, EOFError):
        app.logger.exception("Could not load the saved ML model")
        return None


MODEL = load_model()
init_db()


@app.template_filter("inr")
def format_inr(value: Any) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        number = 0.0
    sign = "-" if number < 0 else ""
    number = abs(number)
    whole, decimals = f"{number:.2f}".split(".")
    if len(whole) > 3:
        last_three = whole[-3:]
        rest = whole[:-3]
        chunks = []
        while len(rest) > 2:
            chunks.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            chunks.insert(0, rest)
        whole = ",".join(chunks + [last_three])
    return f"{sign}₹{whole}.{decimals}"


@app.context_processor
def inject_global_values():
    return {"current_year": date.today().year, "categories": CATEGORIES}


@app.route("/")
def dashboard():
    """Show dashboard analytics from user-saved predictions only.

    The synthetic training dataset and offline model-evaluation records are deliberately
    excluded from dashboard KPIs and charts. A new installation therefore starts empty.
    """
    with get_db() as conn:
        all_predictions = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()

    prediction_count = len(all_predictions)
    recent = all_predictions[:6]
    total_estimated_revenue = round(
        sum(float(row["estimated_revenue"]) for row in all_predictions), 2
    )
    average_estimated_revenue = (
        round(total_estimated_revenue / prediction_count, 2)
        if prediction_count else 0.0
    )
    total_predicted_units = sum(int(row["predicted_units"]) for row in all_predictions)
    categories_tested = len({row["product_category"] for row in all_predictions})

    # Monthly trend is based on the time each prediction was saved, not sample sales rows.
    monthly_totals: dict[str, float] = {}
    category_totals: dict[str, float] = {}
    for row in all_predictions:
        saved_month = str(row["prediction_timestamp"])[:7]
        monthly_totals[saved_month] = monthly_totals.get(saved_month, 0.0) + float(row["estimated_revenue"])
        category = str(row["product_category"])
        category_totals[category] = category_totals.get(category, 0.0) + float(row["estimated_revenue"])

    monthly_items = sorted(monthly_totals.items())[-12:]
    monthly_labels = [month for month, _ in monthly_items]
    monthly_sales = [round(value, 2) for _, value in monthly_items]

    category_items = sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
    category_labels = [category for category, _ in category_items]
    category_sales = [round(value, 2) for _, value in category_items]

    # Show the latest ten saved forecasts in chronological order for the chart.
    chart_predictions = list(reversed(all_predictions[:10]))
    forecast_labels = [
        f"{str(row['product_category'])[:8]} #{row['id']}" for row in chart_predictions
    ]
    forecast_units = [int(row["predicted_units"]) for row in chart_predictions]

    return render_template(
        "dashboard.html",
        total_estimated_revenue=total_estimated_revenue,
        prediction_count=prediction_count,
        average_estimated_revenue=average_estimated_revenue,
        total_predicted_units=total_predicted_units,
        categories_tested=categories_tested,
        monthly_labels=monthly_labels,
        monthly_sales=monthly_sales,
        category_labels=category_labels,
        category_sales=category_sales,
        forecast_labels=forecast_labels,
        forecast_units=forecast_units,
        recent=recent,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    form_values = {
        "prediction_date": date.today().isoformat(),
        "product_category": "Electronics",
        "unit_price": "1500",
        "discount_pct": "10",
        "marketing_spend": "1000",
    }
    result = None

    if request.method == "POST":
        form_values = {key: request.form.get(key, "").strip() for key in form_values}
        try:
            selected_date = date.fromisoformat(form_values["prediction_date"])
            category = form_values["product_category"]
            unit_price = float(form_values["unit_price"])
            discount_pct = float(form_values["discount_pct"])
            marketing_spend = float(form_values["marketing_spend"])

            if category not in CATEGORIES:
                raise ValueError("Please select a valid product category.")
            if not np.isfinite(unit_price) or unit_price <= 0 or unit_price > 10_000_000:
                raise ValueError("Unit price must be greater than ₹0 and within a reasonable range.")
            if not np.isfinite(discount_pct) or not 0 <= discount_pct <= 100:
                raise ValueError("Discount must be between 0% and 100%.")
            if not np.isfinite(marketing_spend) or marketing_spend < 0 or marketing_spend > 100_000_000:
                raise ValueError("Marketing spend cannot be negative or exceed ₹10 crore.")
            if MODEL is None:
                raise RuntimeError("The ML model is not available. Run `python generate_data.py` and `python train_model.py`, then restart Flask.")

            features = pd.DataFrame([{
                "product_category": category,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "marketing_spend": marketing_spend,
                "day_of_week": selected_date.weekday(),
                "month": selected_date.month,
            }])
            raw_prediction = float(MODEL.predict(features)[0])
            predicted_units = max(0, int(round(raw_prediction)))
            estimated_revenue = round(predicted_units * unit_price * (1 - discount_pct / 100), 2)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with get_db() as conn:
                cursor = conn.execute("""
                    INSERT INTO predictions (
                        prediction_timestamp, selected_date, product_category,
                        unit_price, discount_pct, marketing_spend,
                        predicted_units, estimated_revenue
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, selected_date.isoformat(), category, unit_price,
                      discount_pct, marketing_spend, predicted_units, estimated_revenue))
                prediction_id = cursor.lastrowid
                conn.commit()
            flash("Prediction created and saved to your history.", "success")
            return redirect(url_for("predict", prediction_id=prediction_id))
        except (ValueError, TypeError) as exc:
            flash(str(exc) or "Please enter valid values in all fields.", "error")
        except RuntimeError as exc:
            flash(str(exc), "error")
        except Exception:
            app.logger.exception("Prediction failed")
            flash("Prediction could not be completed. Check that the saved model and dependencies are available.", "error")

    prediction_id = request.args.get("prediction_id", type=int)
    if prediction_id:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
            if row:
                result = dict(row)
    return render_template("predict.html", form_values=form_values, result=result)


@app.route("/history")
def history():
    with get_db() as conn:
        records = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    return render_template("history.html", records=records, record_count=len(records))


@app.route("/download/sample-data")
def download_sample_data():
    if not DATA_FILE.exists():
        flash("Sample dataset is missing. Run `python generate_data.py` first.", "error")
        return redirect(url_for("dashboard"))
    return Response(
        DATA_FILE.read_bytes(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sales_data.csv"},
    )


@app.route("/download/predictions")
def download_predictions():
    with get_db() as conn:
        records = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    columns = ["id", "prediction_timestamp", "selected_date", "product_category", "unit_price", "discount_pct", "marketing_spend", "predicted_units", "estimated_revenue"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    for record in records:
        writer.writerow([record[column] for column in columns])
    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


@app.route("/about")
def about():
    return render_template("about.html", metrics=load_metrics())


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    print("\nAI-Based Sales Prediction and Business Analytics System")
    print("Open http://127.0.0.1:5000 in your browser. Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
