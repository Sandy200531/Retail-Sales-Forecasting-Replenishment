# Retail Sales Forecasting & Replenishment

A complete, reproducible portfolio project using a **synthetic** retailer dataset (not real company data).

## Business problem
Forecast daily product/store demand from two years of sales, promotion and inventory data; compare a simple baseline with machine-learning regression; report errors by category; and translate forecasts into suggested reorder quantities.

## Dataset
- Date range: 2024-01-01 to 2025-12-31 (730 days)
- 4 stores × 8 products × 730 days = 23,360 product-store-day rows
- Categories: Beverages, Snacks, Household, Personal Care
- Includes price, promotion flag, discount, receipts, available stock, units sold, ending stock and stockout indicator.
- `retail_daily_sales_raw.csv` intentionally contains a small number of missing and invalid values for cleaning practice.
- `retail_daily_sales_clean.csv` is the cleaned counterpart.

## Quick start
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/retail_sales_forecasting_eda.ipynb
```

Or run the end-to-end script:

```bash
python src/run_pipeline.py
```

Outputs are written to `outputs/` and `reports/`.

## Method
- Time-based split: train through 2025-09-30; test on 2025-10-01 through 2025-12-31.
- Features: calendar, weekend, promotion/discount, price, lagged sales (1, 7, 14, 28 days), rolling means (7, 28 days).
- Baseline: seasonal-naive (same product/store sales 7 days earlier).
- Models: RandomForestRegressor baseline configuration and tuned RandomForestRegressor using a small time-series validation search.
- Metrics: MAE, RMSE, WAPE and bias; also MAE/WAPE by category.
- Reorder suggestion: forecast lead-time demand + safety stock − current stock position, floored at zero.

## Important limitations
This is a synthetic demonstration. It is not evidence of real retail performance. The generated demand process is simplified; stockouts censor observed sales; the data does not contain supplier lead times, purchase orders, returns, local events, competitor prices or actual costs. Reorder outputs use an assumed 7-day lead time and 20% safety-stock buffer. Validate with real data and operational constraints before purchasing inventory.
