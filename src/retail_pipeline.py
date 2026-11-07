
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATE_COL = "date"
GROUP_COLS = ["store_id", "product_id"]

def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean missing/invalid values without using future target values."""
    df = frame.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, "store_id", "product_id"])
    for col in ["unit_price", "discount_pct", "promotion", "stock_received",
                "stock_available", "ending_stock", "stockout", "units_sold"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_price"] = df.groupby("product_id")["unit_price"].transform(lambda s: s.fillna(s.median()))
    df["units_sold"] = df["units_sold"].clip(lower=0)
    df["units_sold"] = df.groupby(GROUP_COLS)["units_sold"].transform(lambda s: s.fillna(s.median()))
    for col in ["stock_received", "stock_available", "ending_stock", "promotion", "stockout"]:
        df[col] = df[col].fillna(0).clip(lower=0)
    df["discount_pct"] = df["discount_pct"].fillna(0).clip(0, 1)
    df["units_sold"] = df["units_sold"].fillna(0).round().astype(int)
    df["promotion"] = df["promotion"].astype(int)
    return df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create strictly backward-looking lags and rolling statistics."""
    df = frame.sort_values(GROUP_COLS + [DATE_COL]).copy()
    dt = df[DATE_COL]
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_year_sin"] = np.sin(2*np.pi*dt.dt.dayofyear/365.25)
    df["day_of_year_cos"] = np.cos(2*np.pi*dt.dt.dayofyear/365.25)
    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = df.groupby(GROUP_COLS)["units_sold"].shift(lag)
    for window in [7, 28]:
        df[f"sales_roll_mean_{window}"] = (
            df.groupby(GROUP_COLS)["units_sold"].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).mean()
            )
        )
    return df

def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(np.asarray(y_true)-np.asarray(y_pred))) / denom) if denom else float("nan")

def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE": wape(y_true, y_pred),
        "Bias": float(np.mean(np.asarray(y_pred)-np.asarray(y_true))),
    }

def run(input_csv, output_dir="outputs", report_dir="reports"):
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_csv)
    data = clean_data(raw)
    featured = add_features(data)
    feature_cols = [
        "unit_price", "discount_pct", "promotion", "day_of_week", "day_of_month",
        "month", "week_of_year", "is_weekend", "day_of_year_sin", "day_of_year_cos",
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_roll_mean_7", "sales_roll_mean_28"
    ]
    model_data = featured.dropna(subset=feature_cols).copy()
    train = model_data[model_data[DATE_COL] <= "2025-06-30"]
    valid = model_data[(model_data[DATE_COL] >= "2025-07-01") & (model_data[DATE_COL] <= "2025-09-30")]
    test = model_data[model_data[DATE_COL] >= "2025-10-01"]
    X_train, y_train = train[feature_cols], train["units_sold"]
    X_valid, y_valid = valid[feature_cols], valid["units_sold"]
    X_test, y_test = test[feature_cols], test["units_sold"]

    # Seasonal-naive baseline uses last week's observed sales.
    baseline_pred = test["sales_lag_7"].to_numpy()
    base_metrics = evaluate(y_test, baseline_pred)

    baseline_model = RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    baseline_model.fit(X_train, y_train)
    rf_pred = baseline_model.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred)

    # Small validation-only hyperparameter search (no test leakage).
    candidates = [
        {"n_estimators": 160, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.8},
        {"n_estimators": 220, "max_depth": 14, "min_samples_leaf": 2, "max_features": 0.8},
        {"n_estimators": 220, "max_depth": 18, "min_samples_leaf": 3, "max_features": 1.0},
    ]
    best, best_mae = None, float("inf")
    for params in candidates:
        candidate = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        candidate.fit(X_train, y_train)
        score = mean_absolute_error(y_valid, candidate.predict(X_valid))
        if score < best_mae:
            best, best_mae = params, score
    tuned = RandomForestRegressor(**best, random_state=42, n_jobs=-1)
    tuned.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
    tuned_pred = np.maximum(0, tuned.predict(X_test))
    tuned_metrics = evaluate(y_test, tuned_pred)

    comparison = pd.DataFrame([
        {"Model":"Seasonal naive (7-day lag)", **base_metrics},
        {"Model":"Random Forest baseline", **rf_metrics},
        {"Model":"Random Forest tuned", **tuned_metrics},
    ])
    comparison.to_csv(report_dir/"model_comparison.csv", index=False)

    pred = test[["date","store_id","product_id","category","product_name","units_sold","stock_available","ending_stock","stockout"]].copy()
    pred["seasonal_naive_prediction"] = baseline_pred
    pred["rf_baseline_prediction"] = np.maximum(0, rf_pred)
    pred["forecast_units"] = tuned_pred
    pred.to_csv(output_dir/"test_forecasts.csv", index=False)

    category_rows=[]
    for category, part in pred.groupby("category"):
        category_rows.append({"category":category, **evaluate(part["units_sold"], part["forecast_units"])})
    category_metrics = pd.DataFrame(category_rows).sort_values("MAE")
    category_metrics.to_csv(report_dir/"category_error_report.csv", index=False)

    # Refit final tuned model on all available historical feature rows.
    final_model = RandomForestRegressor(**best, random_state=42, n_jobs=-1)
    final_model.fit(model_data[feature_cols], model_data["units_sold"])
    # Create a 7-day forward forecast recursively per store/SKU.
    forecast_rows=[]
    history = data.sort_values(GROUP_COLS+[DATE_COL]).copy()
    last_date = history[DATE_COL].max()
    for (store, product), grp in history.groupby(GROUP_COLS):
        grp = grp.sort_values(DATE_COL).copy()
        meta = grp.iloc[-1]
        series = list(grp["units_sold"].astype(float).values)
        for step in range(1,8):
            d = last_date + pd.Timedelta(days=step)
            row = {
                "unit_price": float(meta["unit_price"]),
                "discount_pct": 0.0, "promotion": 0,
                "day_of_week": d.dayofweek, "day_of_month": d.day,
                "month": d.month, "week_of_year": int(d.isocalendar().week),
                "is_weekend": int(d.dayofweek>=5),
                "day_of_year_sin": np.sin(2*np.pi*d.dayofyear/365.25),
                "day_of_year_cos": np.cos(2*np.pi*d.dayofyear/365.25),
                "sales_lag_1": series[-1], "sales_lag_7": series[-7],
                "sales_lag_14": series[-14], "sales_lag_28": series[-28],
                "sales_roll_mean_7": float(np.mean(series[-7:])),
                "sales_roll_mean_28": float(np.mean(series[-28:])),
            }
            pred_units = max(0.0, float(final_model.predict(pd.DataFrame([row])[feature_cols])[0]))
            series.append(pred_units)
            forecast_rows.append({
                "date":d.date().isoformat(),"store_id":store,"product_id":product,
                "category":meta["category"],"product_name":meta["product_name"],
                "forecast_units":round(pred_units,2)
            })
    future = pd.DataFrame(forecast_rows)
    future.to_csv(output_dir/"next_7_day_forecast.csv", index=False)

    # Reorder recommendation: 7-day forecast + 20% safety buffer - current on-hand stock.
    stock = history.groupby(GROUP_COLS).tail(1)[GROUP_COLS+["ending_stock","category","product_name"]]
    demand7 = future.groupby(GROUP_COLS)["forecast_units"].sum().reset_index(name="forecast_7d")
    reorder = stock.merge(demand7, on=GROUP_COLS)
    reorder["safety_stock"] = np.ceil(reorder["forecast_7d"]*0.20)
    reorder["recommended_reorder_qty"] = np.maximum(
        0, np.ceil(reorder["forecast_7d"] + reorder["safety_stock"] - reorder["ending_stock"])
    ).astype(int)
    reorder.to_csv(output_dir/"reorder_recommendations.csv", index=False)

    # Text evaluation report.
    report = [
        "# Model evaluation report", "",
        "Test period: 2025-10-01 to 2025-12-31. Forecasts are one-step predictions using lagged observed sales.",
        "The tuned model is selected using validation data (2025-07-01 to 2025-09-30), then refit on train + validation.",
        "", "## Overall comparison", comparison.to_markdown(index=False), "",
        "## Tuned model error by category", category_metrics.to_markdown(index=False), "",
        "## Selected tuned parameters", f"`{best}`", "",
        "## Interpretation", "MAE is average absolute units of error; RMSE penalizes larger misses; WAPE scales total absolute error by total actual units; positive bias means over-forecasting.",
        "", "## Caveats", "Observed sales may understate demand during stockouts. The synthetic data and simplified replenishment process are illustrative only."
    ]
    (report_dir/"evaluation_report.md").write_text("\\n".join(report), encoding="utf-8")
    return comparison, category_metrics, future, reorder

if __name__ == "__main__":
    run("data/retail_daily_sales_raw.csv")
