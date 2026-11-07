# Retail Sales Forecasting & Replenishment

## Project Overview

This project focuses on analyzing retail sales data, forecasting future product demand, evaluating machine learning models, and recommending inventory reorder quantities.

The objective is to help retailers make better inventory management decisions by understanding historical sales patterns, promotional activities, product performance, and stock availability.

**Note:** The current implementation uses a Python-generated synthetic dataset covering two years (2024–2025). It is designed to demonstrate the complete machine learning workflow and does not represent actual retailer data.

## Business Problem

Retail businesses need to maintain sufficient inventory to meet customer demand while avoiding overstocking and unnecessary storage costs.

This project addresses the problem by:
- Analyzing historical daily product and store sales.
- Identifying sales trends and seasonal patterns.
- Understanding the impact of promotions and discounts.
- Forecasting future product demand.
- Comparing machine learning models.
- Recommending reorder quantities based on forecast demand and available stock.

## Dataset Description

The project uses a synthetic retail sales dataset generated using Python.

**Dataset details:**
- Date range: January 2024 – December 2025
- Number of stores: 4
- Number of products: 8
- Number of categories: 4
- Total records: 23,360

**Product categories:**
- Beverages
- Snacks
- Household
- Personal Care

### Dataset Features

| Column | Description |
|---|---|
| date | Daily transaction date |
| store_id | Store identifier |
| product_id | Product identifier |
| category | Product category |
| product_name | Product name |
| unit_price | Selling price |
| discount_pct | Discount percentage |
| promotion | Promotion indicator |
| stock_received | Units received |
| stock_available | Available stock |
| units_sold | Units sold |
| ending_stock | Remaining inventory |
| stockout | Stockout indicator |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Analyzed daily sales trends.
- Examined product and category performance.
- Identified weekday and seasonal sales patterns.
- Analyzed promotions and discounts.

### 2. Data Preprocessing

- Handled missing values.
- Removed invalid negative sales values.
- Converted date columns into datetime format.
- Corrected data types.
- Prepared data for model training.

### 3. Feature Engineering

Created time-based and historical sales features:

- Day of week
- Day of month
- Month
- Weekend indicator
- Seasonal date features
- 1-day, 7-day, 14-day and 28-day sales lags
- 7-day and 28-day rolling averages

### 4. Model Development

Implemented and compared:

- Seasonal Naive Forecasting
- Random Forest Regression (Baseline)
- Tuned Random Forest Regression

Used chronological training, validation and testing splits to avoid random time-series data leakage.

### 5. Model Evaluation

Evaluated models using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Weighted Absolute Percentage Error (WAPE)
- Forecast Bias

Also evaluated forecasting errors separately for each product category.

### 6. Inventory Reorder Recommendations

Generated reorder recommendations using:
- Forecasted 7-day demand.
- Current ending stock.
- An assumed 7-day supplier lead time.
- A 20% safety stock buffer.

The recommended reorder quantity is calculated as:

Reorder Quantity = Forecast Demand + Safety Stock − Available Inventory

The final quantity is rounded up and cannot be negative.

## Model Performance

Results from the synthetic test dataset:

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal Naive | 9.52 | 15.57 |
| Random Forest Baseline | 5.18 | 8.45 |
| Tuned Random Forest | 5.13 | 8.36 |

The tuned Random Forest achieved the lowest MAE and RMSE among the evaluated models.

## Project Structure

```text
retail_sales_forecasting_project/
│
├── data/
│   ├── retail_daily_sales_raw.csv
│   ├── retail_daily_sales_clean.csv
│   └── DATA_DICTIONARY.md
│
├── notebooks/
│   └── retail_sales_forecasting_eda.ipynb
│
├── src/
│   ├── retail_pipeline.py
│   └── run_pipeline.py
│
├── outputs/
│   ├── test_forecasts.csv
│   ├── next_7_day_forecast.csv
│   └── reorder_recommendations.csv
│
├── reports/
│   ├── evaluation_report.md
│   ├── model_comparison.csv
│   ├── category_error_report.csv
│   └── run_summary.md
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### Step 1: Clone or download the project

Extract the project ZIP file and open the folder in VS Code.

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

If required, install tabulate separately:

```bash
python -m pip install tabulate
```

### Step 3: Run the Jupyter Notebook

Open:

```text
notebooks/retail_sales_forecasting_eda.ipynb
```

Run all cells sequentially.

### Step 4: Run the complete pipeline

From the project root directory, execute:

```bash
python src/run_pipeline.py
```

The generated forecasts, evaluation reports and reorder recommendations will be saved in the respective folders.

## Expected Deliverables

1. Exploratory Data Analysis notebook.
2. Data cleaning and preprocessing pipeline.
3. Time-based feature engineering.
4. Baseline forecasting model.
5. Tuned machine learning model.
6. Model comparison and evaluation report.
7. Future sales forecast output.
8. Product category-wise error analysis.
9. Inventory reorder recommendations.
10. Explanation of project limitations.

## Limitations

- The dataset is synthetic and does not represent actual retailer transactions.
- Sales during stockouts may underestimate actual customer demand.
- Supplier lead time is assumed to be seven days.
- Safety stock is calculated using an assumed 20% buffer.
- External factors such as weather, competitor prices and local events are not included.
- The future forecast is recursive, so prediction errors can accumulate.
- Real-world deployment requires validation using actual historical sales and inventory records.

## Future Enhancements

- Integrate real-world retail datasets such as the Kaggle M5 Walmart dataset.
- Experiment with XGBoost and LightGBM.
- Incorporate actual supplier lead times.
- Develop an interactive sales forecasting dashboard.
- Improve inventory optimization using service-level targets.
- Deploy the forecasting system as a web application.

## Conclusion

This project demonstrates an end-to-end retail sales forecasting and inventory replenishment workflow using Python and machine learning.

By analyzing historical sales, engineering time-based features, comparing predictive models and generating reorder recommendations, the system provides a foundation for data-driven retail inventory management.

---

**Project Domain:** Retail Analytics | Time Series Forecasting | Machine Learning | Inventory Management

**Developed using:** Python, Pandas, NumPy and Scikit-learn
