-- Mirror statistics at HS6 x month: what Korea says it exported to the US (Korea Customs Service, FOB)
-- against what the US says it imported from Korea (Census, customs value). HS6 is the most detailed level
-- shared by both national tariff schedules (Korea's HSK10 and the US HTS10 split HS6 differently).
-- `us_published` is a month-level flag: a small HS6 line with no US value in a published month is a real
-- zero on the US side, not a missing month, so it must stay in the Korean total for that month.
with kr as (
    select month, hs6, sum(export_usd) as kr_export_usd, sum(export_kg) as kr_export_kg
    from {{ ref('stg_kcs__kr_exports_us') }}
    group by month, hs6
),
us as (
    select month, left(hs10, 6) as hs6,
           sum(value_usd) as us_import_usd,
           sum(vessel_value_usd) as us_vessel_usd, sum(air_value_usd) as us_air_usd,
           sum(other_value_usd) as us_other_usd,
           sum(vessel_weight_kg) as us_vessel_kg, sum(air_weight_kg) as us_air_kg
    from {{ ref('stg_census__hs10_monthly') }}
    group by month, left(hs10, 6)
),
us_months as (select distinct month from us),
joined as (
    select
        coalesce(kr.month, us.month)                               as month,
        coalesce(kr.hs6, us.hs6)                                   as hs6,
        coalesce(kr.kr_export_usd, 0)                              as kr_export_usd,
        coalesce(kr.kr_export_kg, 0)                               as kr_export_kg,
        us.us_import_usd, us.us_vessel_usd, us.us_air_usd, us.us_other_usd, us.us_vessel_kg, us.us_air_kg
    from kr
    full outer join us on us.month = kr.month and us.hs6 = kr.hs6
)
select
    j.month,
    j.hs6,
    strftime(j.month, '%Y-%m') || '|' || j.hs6                    as grain_key,
    j.kr_export_usd,
    j.kr_export_kg,
    case when um.month is not null then coalesce(j.us_import_usd, 0) end as us_import_usd,
    case when um.month is not null then coalesce(j.us_vessel_usd, 0) end as us_vessel_usd,
    case when um.month is not null then coalesce(j.us_air_usd, 0) end    as us_air_usd,
    case when um.month is not null then coalesce(j.us_other_usd, 0) end  as us_other_usd,
    case when um.month is not null then coalesce(j.us_vessel_kg, 0) end  as us_vessel_kg,
    case when um.month is not null then coalesce(j.us_air_kg, 0) end     as us_air_kg,
    um.month is not null                                           as us_published
from joined j
left join us_months um on um.month = j.month
