# EDA summary

- Training rows: 400
- Target (default) rate: 6.50%
- Features: 132 (111 numeric, 21 categorical)

## Missingness (top 15)

No history in a source shows up as an explicit `has_*_history` flag rather than nulls in the count columns (see the dbt mart) - the nulls below are genuine missing values within existing records.

| feature | missing rate |
|---|---|
| `cc_amt_balance_avg` | 78.0% |
| `cc_dpd_max` | 78.0% |
| `cc_dpd_avg` | 78.0% |
| `cc_min_payment_ratio_avg` | 78.0% |
| `cc_dpd_def_max` | 78.0% |
| `cc_utilization_max` | 78.0% |
| `cc_utilization_avg` | 78.0% |
| `cc_months_balance_min` | 78.0% |
| `cc_amt_drawings_atm_avg` | 78.0% |
| `cc_amt_balance_max` | 78.0% |
| `cc_amt_payment_total_avg` | 78.0% |
| `cc_amt_drawings_avg` | 78.0% |
| `cc_credit_limit_avg` | 78.0% |
| `prev_app_down_payment_rate_avg` | 65.0% |
| `own_car_age` | 58.2% |

## Numeric features most correlated with TARGET (top 15 by absolute correlation)

| feature | correlation |
|---|---|
| `cc_months_balance_min` | -0.179 |
| `bureau_credit_day_overdue_max` | +0.175 |
| `reg_region_not_work_region` | +0.132 |
| `prev_app_down_payment_rate_avg` | +0.128 |
| `cc_utilization_max` | +0.123 |
| `region_rating_client_w_city` | -0.114 |
| `pos_dpd_rate` | +0.112 |
| `pos_months_balance_min` | +0.102 |
| `total_area_mode` | +0.090 |
| `ext_source_3` | -0.089 |
| `documents_submitted_count` | +0.086 |
| `pos_records_count` | -0.084 |
| `pos_active_count` | -0.083 |
| `def_30_cnt_social_circle` | +0.082 |
| `prev_app_credit_to_application_ratio_avg` | +0.082 |

## Categorical features with the widest target-rate spread (top 10)

Spread = (highest category default rate) - (lowest), restricted to categories with at least 20 applicants so a single-applicant outlier category can't dominate.

| feature | rate spread | categories |
|---|---|---|
| `apply_weekday` | 7.1% | 5 |
| `income_type` | 7.0% | 4 |
| `family_status` | 4.5% | 4 |
| `is_pensioner_anomaly` | 4.5% | 2 |
| `walls_material` | 4.1% | 3 |
| `has_pos_cash_history` | 3.4% | 2 |
| `has_credit_card_history` | 3.3% | 2 |
| `has_bureau_history` | 3.3% | 2 |
| `own_realty` | 3.2% | 2 |
| `organization_type` | 3.1% | 4 |
