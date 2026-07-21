-- The star of the pipeline: one row per SK_ID_CURR, joining the applicant
-- record to every aggregated slice of one-to-many history. This is the only
-- table the modeling layer reads from.
--
-- Count-style columns are coalesced to 0 when an applicant has no history in
-- a given source (that's a true zero, not missing data). Ratio/average
-- columns are left NULL when there's no history to average over - LightGBM
-- and XGBoost both split on missingness natively, so an explicit sentinel
-- would only destroy information. An explicit has_*_history flag is added
-- per source so "no history" is still directly queryable/explainable
-- (and shows up as its own SHAP feature) instead of being implicit in nulls.

select
    app.*,

    coalesce(bureau.bureau_count, 0)                          as bureau_count,
    bureau.bureau_credit_types_count,
    coalesce(bureau.bureau_active_count, 0)                   as bureau_active_count,
    coalesce(bureau.bureau_closed_count, 0)                   as bureau_closed_count,
    bureau.bureau_days_since_credit_applied_avg,
    bureau.bureau_days_since_credit_applied_min,
    bureau.bureau_credit_day_overdue_max,
    coalesce(bureau.bureau_cnt_credit_prolong_sum, 0)         as bureau_cnt_credit_prolong_sum,
    bureau.bureau_amt_credit_sum_total,
    bureau.bureau_amt_credit_sum_debt_total,
    bureau.bureau_amt_credit_sum_overdue_total,
    bureau.bureau_debt_to_credit_ratio,
    bureau.bureau_overdue_to_credit_ratio,
    bureau.bureau_bb_months_count_avg,
    bureau.bureau_bb_dpd_rate_avg,
    bureau.bureau_bb_max_dpd_status_max,
    (bureau.sk_id_curr is not null)                           as has_bureau_history,

    coalesce(prev.prev_app_count, 0)                          as prev_app_count,
    coalesce(prev.prev_app_approved_count, 0)                 as prev_app_approved_count,
    coalesce(prev.prev_app_refused_count, 0)                  as prev_app_refused_count,
    coalesce(prev.prev_app_canceled_count, 0)                 as prev_app_canceled_count,
    prev.prev_app_approval_rate,
    prev.prev_app_amt_credit_avg,
    prev.prev_app_amt_annuity_avg,
    prev.prev_app_amt_application_avg,
    prev.prev_app_credit_to_application_ratio_avg,
    prev.prev_app_down_payment_rate_avg,
    prev.prev_app_cnt_payment_avg,
    prev.prev_app_days_since_decision_min,
    prev.prev_app_days_since_decision_max,
    coalesce(prev.prev_app_new_client_count, 0)               as prev_app_new_client_count,
    (prev.sk_id_curr is not null)                             as has_previous_application_history,

    coalesce(pos.pos_records_count, 0)                        as pos_records_count,
    coalesce(pos.pos_distinct_prev_count, 0)                  as pos_distinct_prev_count,
    coalesce(pos.pos_completed_count, 0)                      as pos_completed_count,
    coalesce(pos.pos_active_count, 0)                         as pos_active_count,
    pos.pos_cnt_instalment_avg,
    pos.pos_months_balance_min,
    pos.pos_dpd_max,
    pos.pos_dpd_avg,
    pos.pos_dpd_def_max,
    pos.pos_dpd_rate,
    (pos.sk_id_curr is not null)                              as has_pos_cash_history,

    coalesce(cc.cc_records_count, 0)                          as cc_records_count,
    coalesce(cc.cc_distinct_prev_count, 0)                    as cc_distinct_prev_count,
    cc.cc_amt_balance_avg,
    cc.cc_amt_balance_max,
    cc.cc_credit_limit_avg,
    cc.cc_utilization_avg,
    cc.cc_utilization_max,
    cc.cc_amt_drawings_atm_avg,
    cc.cc_amt_drawings_avg,
    cc.cc_amt_payment_total_avg,
    cc.cc_min_payment_ratio_avg,
    coalesce(cc.cc_underpaid_months_count, 0)                 as cc_underpaid_months_count,
    cc.cc_months_balance_min,
    cc.cc_dpd_max,
    cc.cc_dpd_avg,
    cc.cc_dpd_def_max,
    (cc.sk_id_curr is not null)                               as has_credit_card_history,

    coalesce(inst.inst_payments_count, 0)                     as inst_payments_count,
    coalesce(inst.inst_distinct_prev_count, 0)                as inst_distinct_prev_count,
    inst.inst_days_late_avg,
    inst.inst_days_late_max,
    inst.inst_late_payment_rate,
    inst.inst_severely_late_payment_rate,
    inst.inst_payment_ratio_avg,
    inst.inst_underpayment_rate,
    inst.inst_amt_instalment_sum,
    inst.inst_amt_payment_sum,
    (inst.sk_id_curr is not null)                             as has_installments_history

from {{ ref('stg_application') }} app
left join {{ ref('int_bureau_agg') }} bureau
    on app.sk_id_curr = bureau.sk_id_curr
left join {{ ref('int_previous_application_agg') }} prev
    on app.sk_id_curr = prev.sk_id_curr
left join {{ ref('int_pos_cash_agg') }} pos
    on app.sk_id_curr = pos.sk_id_curr
left join {{ ref('int_credit_card_agg') }} cc
    on app.sk_id_curr = cc.sk_id_curr
left join {{ ref('int_installments_agg') }} inst
    on app.sk_id_curr = inst.sk_id_curr
