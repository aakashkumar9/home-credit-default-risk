-- Aggregates stg_pos_cash_balance from (sk_id_prev, months_balance) grain up to
-- sk_id_curr grain - one row per applicant summarizing their POS/cash loan history.

select
    sk_id_curr,
    count(*)                                                          as pos_records_count,
    count(distinct sk_id_prev)                                        as pos_distinct_prev_count,
    sum(case when contract_status = 'Completed' then 1 else 0 end)   as pos_completed_count,
    sum(case when contract_status = 'Active' then 1 else 0 end)      as pos_active_count,
    avg(cnt_instalment)                                               as pos_cnt_instalment_avg,
    min(months_balance)                                               as pos_months_balance_min,
    max(dpd)                                                          as pos_dpd_max,
    avg(dpd)                                                          as pos_dpd_avg,
    max(dpd_def)                                                      as pos_dpd_def_max,
    avg(case when dpd > 0 then 1.0 else 0.0 end)                     as pos_dpd_rate
from {{ ref('stg_pos_cash_balance') }}
group by 1
