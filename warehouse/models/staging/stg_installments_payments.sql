-- Grain: (SK_ID_PREV, NUM_INSTALMENT_NUMBER, NUM_INSTALMENT_VERSION). Repayment history.

select
    "SK_ID_PREV"                as sk_id_prev,
    "SK_ID_CURR"                as sk_id_curr,
    "NUM_INSTALMENT_VERSION"    as instalment_version,
    "NUM_INSTALMENT_NUMBER"     as instalment_number,
    -"DAYS_INSTALMENT"          as days_since_instalment_due,
    -"DAYS_ENTRY_PAYMENT"       as days_since_payment_entered,
    ("DAYS_ENTRY_PAYMENT" - "DAYS_INSTALMENT")  as days_late,
    "AMT_INSTALMENT"            as amt_instalment,
    "AMT_PAYMENT"               as amt_payment,
    case when "AMT_INSTALMENT" > 0 then "AMT_PAYMENT" / "AMT_INSTALMENT" end as payment_ratio
from {{ source('raw', 'installments_payments') }}
