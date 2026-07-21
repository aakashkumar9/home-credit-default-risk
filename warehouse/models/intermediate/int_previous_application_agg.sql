-- Aggregates stg_previous_application from sk_id_prev grain up to sk_id_curr
-- grain - one row per applicant summarizing their history of prior applications
-- at Home Credit itself (distinct from the credit-bureau history in int_bureau_agg).

select
    sk_id_curr,
    count(*)                                                                  as prev_app_count,
    sum(case when contract_status = 'Approved' then 1 else 0 end)            as prev_app_approved_count,
    sum(case when contract_status = 'Refused' then 1 else 0 end)             as prev_app_refused_count,
    sum(case when contract_status = 'Canceled' then 1 else 0 end)            as prev_app_canceled_count,
    avg(case when contract_status = 'Approved' then 1.0 else 0.0 end)        as prev_app_approval_rate,
    avg(amt_credit)                                                          as prev_app_amt_credit_avg,
    avg(amt_annuity)                                                         as prev_app_amt_annuity_avg,
    avg(amt_application)                                                     as prev_app_amt_application_avg,
    avg(credit_to_application_ratio)                                        as prev_app_credit_to_application_ratio_avg,
    avg(down_payment_rate)                                                   as prev_app_down_payment_rate_avg,
    avg(cnt_payment)                                                         as prev_app_cnt_payment_avg,
    min(days_since_decision)                                                 as prev_app_days_since_decision_min,
    max(days_since_decision)                                                 as prev_app_days_since_decision_max,
    sum(case when client_type = 'New' then 1 else 0 end)                    as prev_app_new_client_count
from {{ ref('stg_previous_application') }}
group by 1
