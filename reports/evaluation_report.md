# Model evaluation report\n\nTest period: 2025-10-01 to 2025-12-31. Forecasts are one-step predictions using lagged observed sales.\nThe tuned model is selected using validation data (2025-07-01 to 2025-09-30), then refit on train + validation.\n\n## Overall comparison\n| Model                      |     MAE |     RMSE |     WAPE |       Bias |
|:---------------------------|--------:|---------:|---------:|-----------:|
| Seasonal naive (7-day lag) | 9.52242 | 15.5683  | 1.42355  | -0.156929  |
| Random Forest baseline     | 5.17581 |  8.4526  | 0.773756 | -0.0126191 |
| Random Forest tuned        | 5.13384 |  8.36286 | 0.767483 |  0.0045886 |\n\n## Tuned model error by category\n| category      |     MAE |    RMSE |     WAPE |       Bias |
|:--------------|--------:|--------:|---------:|-----------:|
| Household     | 4.42644 | 7.13839 | 0.748934 |  0.501247  |
| Snacks        | 4.80492 | 7.65512 | 0.729912 |  0.187661  |
| Personal Care | 5.34849 | 8.43386 | 0.79525  | -0.0927959 |
| Beverages     | 5.95552 | 9.953   | 0.790062 | -0.577758  |\n\n## Selected tuned parameters\n`{'n_estimators': 160, 'max_depth': 10, 'min_samples_leaf': 2, 'max_features': 0.8}`\n\n## Interpretation\nMAE is average absolute units of error; RMSE penalizes larger misses; WAPE scales total absolute error by total actual units; positive bias means over-forecasting.\n\n## Caveats\nObserved sales may understate demand during stockouts. The synthetic data and simplified replenishment process are illustrative only.