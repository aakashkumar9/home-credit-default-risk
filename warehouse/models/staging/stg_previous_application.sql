-- Grain: SK_ID_PREV (one row per previous application at Home Credit).
-- Many-to-one with application on SK_ID_CURR.
-- DAYS_* columns use the same 365243 "not applicable" sentinel as application.DAYS_EMPLOYED.

select
    "SK_ID_PREV"                          as sk_id_prev,
    "SK_ID_CURR"                          as sk_id_curr,
    "NAME_CONTRACT_TYPE"                  as contract_type,
    "NAME_CONTRACT_STATUS"                as contract_status,
    "NAME_CASH_LOAN_PURPOSE"              as cash_loan_purpose,
    "NAME_PAYMENT_TYPE"                   as payment_type,
    "CODE_REJECT_REASON"                  as reject_reason,
    "NAME_CLIENT_TYPE"                    as client_type,
    "NAME_GOODS_CATEGORY"                 as goods_category,
    "NAME_PORTFOLIO"                      as portfolio,
    "NAME_PRODUCT_TYPE"                   as product_type,
    "CHANNEL_TYPE"                        as channel_type,
    "NAME_SELLER_INDUSTRY"                as seller_industry,
    "NAME_YIELD_GROUP"                    as yield_group,
    "PRODUCT_COMBINATION"                 as product_combination,

    "AMT_ANNUITY"                         as amt_annuity,
    "AMT_APPLICATION"                     as amt_application,
    "AMT_CREDIT"                          as amt_credit,
    "AMT_DOWN_PAYMENT"                    as amt_down_payment,
    "AMT_GOODS_PRICE"                     as amt_goods_price,
    case when "AMT_APPLICATION" > 0 then "AMT_CREDIT" / "AMT_APPLICATION" end   as credit_to_application_ratio,
    case when "AMT_CREDIT" > 0 then "AMT_DOWN_PAYMENT" / "AMT_CREDIT" end       as down_payment_rate,
    "RATE_DOWN_PAYMENT"                   as rate_down_payment,
    "RATE_INTEREST_PRIMARY"               as rate_interest_primary,
    "RATE_INTEREST_PRIVILEGED"            as rate_interest_privileged,
    "CNT_PAYMENT"                         as cnt_payment,

    -"DAYS_DECISION"                      as days_since_decision,
    case when "DAYS_FIRST_DRAWING" = 365243 then null else -"DAYS_FIRST_DRAWING" end     as days_since_first_drawing,
    case when "DAYS_FIRST_DUE" = 365243 then null else -"DAYS_FIRST_DUE" end             as days_since_first_due,
    case when "DAYS_LAST_DUE_1ST_VERSION" = 365243 then null else -"DAYS_LAST_DUE_1ST_VERSION" end as days_since_last_due_1st_version,
    case when "DAYS_LAST_DUE" = 365243 then null else -"DAYS_LAST_DUE" end               as days_since_last_due,
    case when "DAYS_TERMINATION" = 365243 then null else -"DAYS_TERMINATION" end         as days_since_termination,

    "FLAG_LAST_APPL_PER_CONTRACT"         as is_last_appl_per_contract,
    "NFLAG_LAST_APPL_IN_DAY"              as is_last_appl_in_day,
    "NFLAG_INSURED_ON_APPROVAL"           as is_insured_on_approval,
    "SELLERPLACE_AREA"                    as sellerplace_area
from {{ source('raw', 'previous_application') }}
