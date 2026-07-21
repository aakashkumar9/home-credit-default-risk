-- Aggregates stg_credit_card_balance from (sk_id_prev, months_balance) grain up
-- to sk_id_curr grain - one row per applicant summarizing their credit card
-- utilization and repayment behaviour.

select
    sk_id_curr,
    count(*)                                       as cc_records_count,
    count(distinct sk_id_prev)                      as cc_distinct_prev_count,
    avg(amt_balance)                                as cc_amt_balance_avg,
    max(amt_balance)                                as cc_amt_balance_max,
    avg(amt_credit_limit_actual)                    as cc_credit_limit_avg,
    avg(utilization)                                as cc_utilization_avg,
    max(utilization)                                as cc_utilization_max,
    avg(amt_drawings_atm_current)                   as cc_amt_drawings_atm_avg,
    avg(amt_drawings_current)                       as cc_amt_drawings_avg,
    avg(amt_payment_total_current)                  as cc_amt_payment_total_avg,
    avg(min_payment_ratio)                          as cc_min_payment_ratio_avg,
    sum(case when min_payment_ratio < 1 then 1 else 0 end) as cc_underpaid_months_count,
    min(months_balance)                             as cc_months_balance_min,
    max(dpd)                                        as cc_dpd_max,
    avg(dpd)                                        as cc_dpd_avg,
    max(dpd_def)                                    as cc_dpd_def_max
from {{ ref('stg_credit_card_balance') }}
group by 1
