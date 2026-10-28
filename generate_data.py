"""Generate a reproducible synthetic sales dataset for the academic demo."""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "sales_data.csv"
CATEGORIES = ["Electronics", "Clothing", "Grocery", "Home & Kitchen", "Beauty & Personal Care"]


def generate_dataset(n_records: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2024-01-01").value // 10**9,
            pd.Timestamp("2026-01-01").value // 10**9,
            size=n_records,
        ), unit="s"
    ).normalize()
    categories = rng.choice(CATEGORIES, size=n_records, p=[0.18, 0.24, 0.22, 0.18, 0.18])

    price_ranges = {
        "Electronics": (700, 18000),
        "Clothing": (250, 3500),
        "Grocery": (30, 900),
        "Home & Kitchen": (250, 7000),
        "Beauty & Personal Care": (100, 2500),
    }
    base_demand = {
        "Electronics": 18,
        "Clothing": 42,
        "Grocery": 75,
        "Home & Kitchen": 28,
        "Beauty & Personal Care": 48,
    }
    prices = np.array([rng.uniform(*price_ranges[c]) for c in categories])
    discounts = np.round(rng.choice([0, 5, 10, 15, 20, 25, 30], size=n_records, p=[.12,.12,.2,.2,.17,.12,.07]), 1)
    marketing = np.round(rng.uniform(100, 12000, size=n_records), 2)
    day_of_week = pd.DatetimeIndex(dates).dayofweek.to_numpy()
    month = pd.DatetimeIndex(dates).month.to_numpy()

    category_base = np.array([base_demand[c] for c in categories], dtype=float)
    # Synthetic demand relationship: category demand, affordability, promotions,
    # marketing, weekend traffic, seasonality, and random noise.
    price_position = np.array([
        (prices[i] - price_ranges[categories[i]][0]) /
        max(1, price_ranges[categories[i]][1] - price_ranges[categories[i]][0])
        for i in range(n_records)
    ])
    weekend_effect = np.where(day_of_week >= 5, 1.14, 1.0)
    seasonal_effect = 1 + 0.12 * np.sin((month - 1) * 2 * np.pi / 12) + np.where(np.isin(month, [10, 11, 12]), .15, 0)
    discount_effect = 1 + discounts * 0.012
    marketing_effect = 0.82 + 0.38 * np.sqrt(marketing / 12000)
    price_effect = 1.20 - 0.48 * price_position
    noise = rng.normal(1.0, 0.16, size=n_records)
    units = np.maximum(1, np.rint(category_base * weekend_effect * seasonal_effect * discount_effect * marketing_effect * price_effect * noise)).astype(int)
    sales = np.round(units * prices * (1 - discounts / 100), 2)

    df = pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(1, n_records + 1)],
        "order_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "product_category": categories,
        "unit_price": np.round(prices, 2),
        "discount_pct": discounts,
        "marketing_spend": marketing,
        "day_of_week": day_of_week.astype(int),
        "month": month.astype(int),
        "units_sold": units,
        "sales": sales,
    })
    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df):,} synthetic sales records: {OUTPUT_FILE}")
    print("Note: these are synthetic demonstration records, not real company sales.")


if __name__ == "__main__":
    main()
