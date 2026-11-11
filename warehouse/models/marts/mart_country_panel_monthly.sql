-- Country x month panel for the cross-country comparison: Korea (treated) and the eight peer countries.
-- Same products (HS 3303/3304/3305/3307), same Census endpoint and definitions for every country.
with korea as (
    select month, '5800' as cty_code, value_usd, vessel_value_usd, air_value_usd, vessel_weight_kg, air_weight_kg
    from {{ ref('stg_census__hs10_monthly') }}
),
peers as (
    select month, cty_code, value_usd, vessel_value_usd, air_value_usd, vessel_weight_kg, air_weight_kg
    from {{ ref('stg_census__peer_hs10_monthly') }}
),
agg as (
    select month, cty_code,
           sum(value_usd) as value_usd, sum(vessel_value_usd) as vessel_value_usd, sum(air_value_usd) as air_value_usd,
           sum(vessel_weight_kg) as vessel_weight_kg, sum(air_weight_kg) as air_weight_kg
    from (select * from korea union all select * from peers)
    group by month, cty_code
)
select
    a.month,
    a.cty_code,
    c.country,
    c.currency,
    c.is_treated,
    a.value_usd,
    a.vessel_value_usd,
    a.air_value_usd,
    a.vessel_weight_kg,
    a.air_weight_kg,
    a.vessel_weight_kg + a.air_weight_kg                          as measured_weight_kg,
    (a.vessel_value_usd + a.air_value_usd) / a.value_usd          as sea_air_share_value,
    a.vessel_value_usd / nullif(a.vessel_weight_kg, 0)            as vessel_usd_per_kg,
    a.air_value_usd / nullif(a.air_weight_kg, 0)                  as air_usd_per_kg,
    fx.lcu_per_usd
from agg a
join {{ ref('country_currency') }} c on c.cty_code = a.cty_code
join {{ ref('stg_fed_board__fx_monthly') }} fx on fx.currency = c.currency and fx.month = a.month
