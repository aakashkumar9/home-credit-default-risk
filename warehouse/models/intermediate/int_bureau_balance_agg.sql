-- Aggregates stg_bureau_balance from (sk_id_bureau, months_balance) grain up to
-- sk_id_bureau grain, so int_bureau_agg can join it 1:1 onto stg_bureau.

select
    sk_id_bureau,
    count(*)                                            as bb_months_count,
    min(months_balance)                                 as bb_months_balance_min,
    avg(case when is_dpd then 1.0 else 0.0 end)         as bb_dpd_rate,
    avg(case when status = 'C' then 1.0 else 0.0 end)   as bb_closed_rate,
    avg(case when status = 'X' then 1.0 else 0.0 end)   as bb_unknown_rate,
    max(status_numeric)                                 as bb_max_dpd_status
from {{ ref('stg_bureau_balance') }}
group by 1
