# Data dictionary

| Field | Meaning |
|---|---|
| date | Calendar date of the store-product record |
| store_id | Synthetic store identifier |
| product_id | Synthetic SKU identifier |
| category / product_name | Product classification and name |
| unit_price | Selling price after any discount |
| discount_pct | Fractional discount (0–1) |
| promotion | 1 when a promotion is active, otherwise 0 |
| stock_received | Units received that day |
| stock_available | Inventory available before sales (including receipts) |
| units_sold | Units sold that day (observed demand; may be censored by stockouts) |
| ending_stock | Inventory remaining after sales |
| stockout | 1 if estimated demand exceeded available inventory |

All records are synthetic and generated for this project.
