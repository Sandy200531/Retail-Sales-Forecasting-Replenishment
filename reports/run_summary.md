# Run summary

Rows in raw dataset: 23,392
Rows in cleaned dataset: 23,392
Stores: 4
Products: 8
Categories: 4
Date range: 2024-01-01 to 2025-12-31

## Model comparison

| Model                      |     MAE |     RMSE |     WAPE |       Bias |
|:---------------------------|--------:|---------:|---------:|-----------:|
| Seasonal naive (7-day lag) | 9.52242 | 15.5683  | 1.42355  | -0.156929  |
| Random Forest baseline     | 5.17581 |  8.4526  | 0.773756 | -0.0126191 |
| Random Forest tuned        | 5.13384 |  8.36286 | 0.767483 |  0.0045886 |

## Category-level tuned model errors

| category      |     MAE |    RMSE |     WAPE |       Bias |
|:--------------|--------:|--------:|---------:|-----------:|
| Household     | 4.42644 | 7.13839 | 0.748934 |  0.501247  |
| Snacks        | 4.80492 | 7.65512 | 0.729912 |  0.187661  |
| Personal Care | 5.34849 | 8.43386 | 0.79525  | -0.0927959 |
| Beverages     | 5.95552 | 9.953   | 0.790062 | -0.577758  |

## Generated deliverables
- EDA notebook
- Raw and cleaned CSV data
- Reusable preprocessing/feature engineering/model pipeline
- Model comparison and category error CSVs
- Evaluation report
- 7-day forecast and reorder recommendations
