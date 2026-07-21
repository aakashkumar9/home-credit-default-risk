-- Unions application_train and application_test into a single applicant-grain
-- staging model. TARGET is NULL for the scoring set; is_train distinguishes
-- the two without needing a second model downstream.
--
-- The ~45 building/apartment columns (APARTMENTS_AVG, BASEMENTAREA_AVG, ...
-- each repeated as _AVG/_MODE/_MEDI) are collapsed into a single quality
-- score and a missingness rate. They are highly correlated with each other,
-- mostly describe the same "how well is this building documented" signal,
-- and their missingness is itself a known predictive feature in this
-- dataset - so we keep that signal without carrying 45 near-duplicate
-- columns through every downstream model.

with train as (
    select *, true as is_train
    from {{ source('raw', 'application_train') }}
),

test as (
    select *, cast(null as bigint) as "TARGET", false as is_train
    from {{ source('raw', 'application_test') }}
),

unioned as (
    select * from train
    union all by name
    select * from test
),

renamed as (
    select
        "SK_ID_CURR"                        as sk_id_curr,
        is_train,
        "TARGET"                            as target,

        -- loan
        "NAME_CONTRACT_TYPE"                as contract_type,
        "AMT_CREDIT"                        as amt_credit,
        "AMT_ANNUITY"                       as amt_annuity,
        "AMT_GOODS_PRICE"                   as amt_goods_price,
        case when "AMT_CREDIT" > 0 then "AMT_ANNUITY" / "AMT_CREDIT" end as annuity_to_credit_ratio,
        case when "AMT_GOODS_PRICE" > 0 then "AMT_CREDIT" / "AMT_GOODS_PRICE" end as credit_to_goods_ratio,

        -- demographics
        "CODE_GENDER"                       as gender,
        "FLAG_OWN_CAR"                      as own_car,
        "FLAG_OWN_REALTY"                   as own_realty,
        "CNT_CHILDREN"                      as cnt_children,
        "CNT_FAM_MEMBERS"                   as cnt_fam_members,
        "NAME_TYPE_SUITE"                   as type_suite,
        "NAME_FAMILY_STATUS"                as family_status,
        "NAME_HOUSING_TYPE"                 as housing_type,
        "NAME_EDUCATION_TYPE"               as education_type,

        -- income / employment
        "AMT_INCOME_TOTAL"                  as amt_income_total,
        case when "AMT_INCOME_TOTAL" > 0 then "AMT_CREDIT" / "AMT_INCOME_TOTAL" end as credit_to_income_ratio,
        case when "AMT_INCOME_TOTAL" > 0 then "AMT_ANNUITY" / "AMT_INCOME_TOTAL" end as annuity_to_income_ratio,
        "NAME_INCOME_TYPE"                  as income_type,
        "OCCUPATION_TYPE"                   as occupation_type,
        "ORGANIZATION_TYPE"                 as organization_type,

        -- ages / tenure, in years, sign-flipped (raw is negative days from today)
        -"DAYS_BIRTH" / 365.25              as age_years,
        case when "DAYS_EMPLOYED" = 365243 then null else -"DAYS_EMPLOYED" / 365.25 end as employed_years,
        case when "DAYS_EMPLOYED" = 365243 then true else false end as is_pensioner_anomaly,
        -"DAYS_REGISTRATION" / 365.25       as registration_years,
        -"DAYS_ID_PUBLISH" / 365.25         as id_publish_years,
        -"DAYS_LAST_PHONE_CHANGE" / 365.25  as last_phone_change_years,
        "OWN_CAR_AGE"                       as own_car_age,

        -- external credit scores (already 0-1 normalized, strongest predictors in this dataset)
        "EXT_SOURCE_1"                      as ext_source_1,
        "EXT_SOURCE_2"                      as ext_source_2,
        "EXT_SOURCE_3"                      as ext_source_3,

        -- region / contact risk flags
        "REGION_POPULATION_RELATIVE"        as region_population_relative,
        "REGION_RATING_CLIENT"              as region_rating_client,
        "REGION_RATING_CLIENT_W_CITY"       as region_rating_client_w_city,
        "REG_REGION_NOT_LIVE_REGION"        as reg_region_not_live_region,
        "REG_REGION_NOT_WORK_REGION"        as reg_region_not_work_region,
        "LIVE_REGION_NOT_WORK_REGION"       as live_region_not_work_region,
        "REG_CITY_NOT_LIVE_CITY"            as reg_city_not_live_city,
        "REG_CITY_NOT_WORK_CITY"            as reg_city_not_work_city,
        "LIVE_CITY_NOT_WORK_CITY"           as live_city_not_work_city,
        "FLAG_MOBIL" + "FLAG_EMP_PHONE" + "FLAG_WORK_PHONE" + "FLAG_CONT_MOBILE" + "FLAG_PHONE" + "FLAG_EMAIL"
                                             as contactability_score,

        -- social circle (defaults among the applicant's social circle - a network signal)
        "OBS_30_CNT_SOCIAL_CIRCLE"          as obs_30_cnt_social_circle,
        "DEF_30_CNT_SOCIAL_CIRCLE"          as def_30_cnt_social_circle,
        "OBS_60_CNT_SOCIAL_CIRCLE"          as obs_60_cnt_social_circle,
        "DEF_60_CNT_SOCIAL_CIRCLE"          as def_60_cnt_social_circle,

        -- document flags, collapsed to a count (individual flags are extremely sparse)
        "FLAG_DOCUMENT_2" + "FLAG_DOCUMENT_3" + "FLAG_DOCUMENT_4" + "FLAG_DOCUMENT_5" +
        "FLAG_DOCUMENT_6" + "FLAG_DOCUMENT_7" + "FLAG_DOCUMENT_8" + "FLAG_DOCUMENT_9" +
        "FLAG_DOCUMENT_10" + "FLAG_DOCUMENT_11" + "FLAG_DOCUMENT_12" + "FLAG_DOCUMENT_13" +
        "FLAG_DOCUMENT_14" + "FLAG_DOCUMENT_15" + "FLAG_DOCUMENT_16" + "FLAG_DOCUMENT_17" +
        "FLAG_DOCUMENT_18" + "FLAG_DOCUMENT_19" + "FLAG_DOCUMENT_20" + "FLAG_DOCUMENT_21"
                                             as documents_submitted_count,

        -- credit bureau enquiries in the year before application
        "AMT_REQ_CREDIT_BUREAU_HOUR"        as bureau_enquiries_hour,
        "AMT_REQ_CREDIT_BUREAU_DAY"         as bureau_enquiries_day,
        "AMT_REQ_CREDIT_BUREAU_WEEK"        as bureau_enquiries_week,
        "AMT_REQ_CREDIT_BUREAU_MON"         as bureau_enquiries_mon,
        "AMT_REQ_CREDIT_BUREAU_QRT"         as bureau_enquiries_qrt,
        "AMT_REQ_CREDIT_BUREAU_YEAR"        as bureau_enquiries_year,

        -- housing/building quality: collapse ~45 *_AVG/_MODE/_MEDI columns into one score + missingness rate
        (
            coalesce("APARTMENTS_AVG", 0) + coalesce("BASEMENTAREA_AVG", 0) +
            coalesce("YEARS_BEGINEXPLUATATION_AVG", 0) + coalesce("YEARS_BUILD_AVG", 0) +
            coalesce("COMMONAREA_AVG", 0) + coalesce("ELEVATORS_AVG", 0) +
            coalesce("ENTRANCES_AVG", 0) + coalesce("FLOORSMAX_AVG", 0) +
            coalesce("FLOORSMIN_AVG", 0) + coalesce("LANDAREA_AVG", 0) +
            coalesce("LIVINGAPARTMENTS_AVG", 0) + coalesce("LIVINGAREA_AVG", 0) +
            coalesce("NONLIVINGAPARTMENTS_AVG", 0) + coalesce("NONLIVINGAREA_AVG", 0)
        ) / 14.0                             as housing_quality_score,
        (
            (case when "APARTMENTS_AVG" is null then 1 else 0 end) +
            (case when "BASEMENTAREA_AVG" is null then 1 else 0 end) +
            (case when "YEARS_BEGINEXPLUATATION_AVG" is null then 1 else 0 end) +
            (case when "YEARS_BUILD_AVG" is null then 1 else 0 end) +
            (case when "COMMONAREA_AVG" is null then 1 else 0 end) +
            (case when "ELEVATORS_AVG" is null then 1 else 0 end) +
            (case when "ENTRANCES_AVG" is null then 1 else 0 end) +
            (case when "FLOORSMAX_AVG" is null then 1 else 0 end) +
            (case when "FLOORSMIN_AVG" is null then 1 else 0 end) +
            (case when "LANDAREA_AVG" is null then 1 else 0 end) +
            (case when "LIVINGAPARTMENTS_AVG" is null then 1 else 0 end) +
            (case when "LIVINGAREA_AVG" is null then 1 else 0 end) +
            (case when "NONLIVINGAPARTMENTS_AVG" is null then 1 else 0 end) +
            (case when "NONLIVINGAREA_AVG" is null then 1 else 0 end)
        ) / 14.0                             as housing_info_missing_rate,
        "TOTALAREA_MODE"                     as total_area_mode,
        "WALLSMATERIAL_MODE"                 as walls_material,
        "HOUSETYPE_MODE"                     as house_type,
        "FONDKAPREMONT_MODE"                 as fond_kapremont,
        "EMERGENCYSTATE_MODE"                as emergency_state,

        "WEEKDAY_APPR_PROCESS_START"        as apply_weekday,
        "HOUR_APPR_PROCESS_START"           as apply_hour

    from unioned
)

select *
from renamed
