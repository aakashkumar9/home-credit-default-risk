-- Grain: (SK_ID_PREV, MONTHS_BALANCE). Monthly snapshot of previous credit card loans.

select
    "SK_ID_PREV"                        as sk_id_prev,
    "SK_ID_CURR"                        as sk_id_curr,
    "MONTHS_BALANCE"                    as months_balance,
    "NAME_CONTRACT_STATUS"              as contract_status,
    "AMT_BALANCE"                       as amt_balance,
    "AMT_CREDIT_LIMIT_ACTUAL"           as amt_credit_limit_actual,
    case when "AMT_CREDIT_LIMIT_ACTUAL" > 0 then "AMT_BALANCE" / "AMT_CREDIT_LIMIT_ACTUAL" end as utilization,
    "AMT_DRAWINGS_ATM_CURRENT"          as amt_drawings_atm_current,
    "AMT_DRAWINGS_CURRENT"              as amt_drawings_current,
    "AMT_DRAWINGS_OTHER_CURRENT"        as amt_drawings_other_current,
    "AMT_DRAWINGS_POS_CURRENT"          as amt_drawings_pos_current,
    "AMT_INST_MIN_REGULARITY"          as amt_inst_min_regularity,
    "AMT_PAYMENT_CURRENT"               as amt_payment_current,
    "AMT_PAYMENT_TOTAL_CURRENT"         as amt_payment_total_current,
    case when "AMT_INST_MIN_REGULARITY" > 0 then "AMT_PAYMENT_TOTAL_CURRENT" / "AMT_INST_MIN_REGULARITY" end as min_payment_ratio,
    "AMT_RECEIVABLE_PRINCIPAL"          as amt_receivable_principal,
    "AMT_RECIVABLE"                     as amt_receivable,
    "AMT_TOTAL_RECEIVABLE"              as amt_total_receivable,
    "CNT_DRAWINGS_ATM_CURRENT"          as cnt_drawings_atm_current,
    "CNT_DRAWINGS_CURRENT"              as cnt_drawings_current,
    "CNT_DRAWINGS_OTHER_CURRENT"        as cnt_drawings_other_current,
    "CNT_DRAWINGS_POS_CURRENT"          as cnt_drawings_pos_current,
    "CNT_INSTALMENT_MATURE_CUM"         as cnt_instalment_mature_cum,
    "SK_DPD"                            as dpd,
    "SK_DPD_DEF"                        as dpd_def
from {{ source('raw', 'credit_card_balance') }}
