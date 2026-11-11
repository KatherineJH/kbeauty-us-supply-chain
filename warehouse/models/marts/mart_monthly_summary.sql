-- One row per month: totals, mode mix, unit values, YoY change and post-policy flag.
with m as (
    select
        month,
        sum(value_usd)                                         as value_usd,
        sum(vessel_value_usd)                                  as vessel_value_usd,
        sum(air_value_usd)                                     as air_value_usd,
        sum(other_value_usd)                                   as other_value_usd,
        sum(vessel_weight_kg)                                  as vessel_weight_kg,
        sum(air_weight_kg)                                     as air_weight_kg
    from {{ ref('stg_census__hs10_monthly') }}
    group by month
),
policy as (
    select min(first_full_month) as post_policy_from from {{ ref('policy_events') }}
)
select
    m.*,
    vessel_weight_kg + air_weight_kg                            as measured_weight_kg,
    air_value_usd / value_usd                                   as air_share_value,
    air_weight_kg / nullif(vessel_weight_kg + air_weight_kg, 0) as air_share_weight,
    vessel_value_usd / nullif(vessel_weight_kg, 0)              as vessel_usd_per_kg,
    air_value_usd / nullif(air_weight_kg, 0)                    as air_usd_per_kg,
    value_usd / lag(value_usd, 12) over (order by month) - 1    as value_yoy,
    (vessel_weight_kg + air_weight_kg)
        / lag(vessel_weight_kg + air_weight_kg, 12) over (order by month) - 1 as weight_yoy,
    m.month >= p.post_policy_from                               as is_post_policy
from m cross join policy p
order by month
