-- Aggregates stg_installments_payments from instalment grain up to sk_id_curr
-- grain - one row per applicant summarizing their instalment repayment
-- discipline (lateness and under/over-payment) across all previous credits.

select
    sk_id_curr,
    count(*)                                                      as inst_payments_count,
    count(distinct sk_id_prev)                                    as inst_distinct_prev_count,
    avg(days_late)                                                as inst_days_late_avg,
    max(days_late)                                                as inst_days_late_max,
    avg(case when days_late > 0 then 1.0 else 0.0 end)           as inst_late_payment_rate,
    avg(case when days_late > 30 then 1.0 else 0.0 end)          as inst_severely_late_payment_rate,
    avg(payment_ratio)                                            as inst_payment_ratio_avg,
    avg(case when payment_ratio < 0.99 then 1.0 else 0.0 end)    as inst_underpayment_rate,
    sum(amt_instalment)                                           as inst_amt_instalment_sum,
    sum(amt_payment)                                              as inst_amt_payment_sum
from {{ ref('stg_installments_payments') }}
group by 1
