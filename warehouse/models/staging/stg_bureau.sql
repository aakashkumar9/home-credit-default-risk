-- Grain: SK_ID_BUREAU (one row per credit reported to the bureau).
-- Many-to-one with application on SK_ID_CURR.

select
    "SK_ID_BUREAU"                as sk_id_bureau,
    "SK_ID_CURR"                  as sk_id_curr,
    "CREDIT_ACTIVE"               as credit_active,
    "CREDIT_CURRENCY"             as credit_currency,
    "CREDIT_TYPE"                 as credit_type,
    -"DAYS_CREDIT"                as days_since_credit_applied,
    "CREDIT_DAY_OVERDUE"          as credit_day_overdue,
    -"DAYS_CREDIT_ENDDATE"        as days_to_credit_enddate,
    -"DAYS_ENDDATE_FACT"          as days_since_credit_closed,
    -"DAYS_CREDIT_UPDATE"         as days_since_bureau_update,
    "CNT_CREDIT_PROLONG"          as cnt_credit_prolong,
    "AMT_CREDIT_MAX_OVERDUE"      as amt_credit_max_overdue,
    "AMT_CREDIT_SUM"              as amt_credit_sum,
    "AMT_CREDIT_SUM_DEBT"         as amt_credit_sum_debt,
    "AMT_CREDIT_SUM_LIMIT"        as amt_credit_sum_limit,
    "AMT_CREDIT_SUM_OVERDUE"      as amt_credit_sum_overdue,
    "AMT_ANNUITY"                 as amt_annuity
from {{ source('raw', 'bureau') }}
