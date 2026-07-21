-- Aggregates stg_bureau (joined 1:1 with int_bureau_balance_agg) from sk_id_bureau
-- grain up to sk_id_curr grain - one row per applicant summarizing their whole
-- credit bureau history.

with bureau as (
    select
        b.*,
        bb.bb_months_count,
        bb.bb_months_balance_min,
        bb.bb_dpd_rate,
        bb.bb_closed_rate,
        bb.bb_max_dpd_status
    from {{ ref('stg_bureau') }} b
    left join {{ ref('int_bureau_balance_agg') }} bb
        on b.sk_id_bureau = bb.sk_id_bureau
)

select
    sk_id_curr,
    count(*)                                                              as bureau_count,
    count(distinct credit_type)                                          as bureau_credit_types_count,
    sum(case when credit_active = 'Active' then 1 else 0 end)            as bureau_active_count,
    sum(case when credit_active = 'Closed' then 1 else 0 end)            as bureau_closed_count,
    avg(days_since_credit_applied)                                       as bureau_days_since_credit_applied_avg,
    min(days_since_credit_applied)                                       as bureau_days_since_credit_applied_min,
    max(credit_day_overdue)                                              as bureau_credit_day_overdue_max,
    sum(cnt_credit_prolong)                                              as bureau_cnt_credit_prolong_sum,
    sum(amt_credit_sum)                                                  as bureau_amt_credit_sum_total,
    sum(amt_credit_sum_debt)                                             as bureau_amt_credit_sum_debt_total,
    sum(amt_credit_sum_overdue)                                          as bureau_amt_credit_sum_overdue_total,
    case when sum(amt_credit_sum) > 0
        then sum(amt_credit_sum_debt) / sum(amt_credit_sum)
    end                                                                   as bureau_debt_to_credit_ratio,
    case when sum(amt_credit_sum) > 0
        then sum(amt_credit_sum_overdue) / sum(amt_credit_sum)
    end                                                                   as bureau_overdue_to_credit_ratio,
    avg(bb_months_count)                                                 as bureau_bb_months_count_avg,
    avg(bb_dpd_rate)                                                     as bureau_bb_dpd_rate_avg,
    max(bb_max_dpd_status)                                               as bureau_bb_max_dpd_status_max
from bureau
group by 1
