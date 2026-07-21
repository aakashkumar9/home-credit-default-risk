-- Grain: (SK_ID_BUREAU, MONTHS_BALANCE). Monthly status snapshot of each bureau credit.
-- STATUS: 'C' = closed, 'X' = unknown, '0' = no DPD, '1'..'5' = increasing DPD buckets.

select
    "SK_ID_BUREAU"     as sk_id_bureau,
    "MONTHS_BALANCE"   as months_balance,
    "STATUS"           as status,
    case
        when "STATUS" in ('1', '2', '3', '4', '5') then true
        else false
    end                as is_dpd,
    case
        when "STATUS" ~ '^[0-9]$' then cast("STATUS" as int)
        else null
    end                as status_numeric
from {{ source('raw', 'bureau_balance') }}
