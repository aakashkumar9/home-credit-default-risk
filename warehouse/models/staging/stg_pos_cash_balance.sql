-- Grain: (SK_ID_PREV, MONTHS_BALANCE). Monthly snapshot of previous POS/cash loans.

select
    "SK_ID_PREV"                as sk_id_prev,
    "SK_ID_CURR"                as sk_id_curr,
    "MONTHS_BALANCE"            as months_balance,
    "NAME_CONTRACT_STATUS"      as contract_status,
    "CNT_INSTALMENT"            as cnt_instalment,
    "CNT_INSTALMENT_FUTURE"     as cnt_instalment_future,
    "SK_DPD"                    as dpd,
    "SK_DPD_DEF"                as dpd_def
from {{ source('raw', 'pos_cash_balance') }}
