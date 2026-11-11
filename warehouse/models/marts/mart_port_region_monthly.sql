-- Month x region x gateway type: where K-beauty enters the US.
select
    f.month,
    d.region,
    any_value(d.region_ko)                                      as region_ko,
    d.gateway_type,
    sum(f.value_usd)                                            as value_usd,
    sum(f.vessel_value_usd)                                     as vessel_value_usd,
    sum(f.air_value_usd)                                        as air_value_usd,
    sum(f.vessel_weight_kg)                                     as vessel_weight_kg,
    sum(f.air_weight_kg)                                        as air_weight_kg
from {{ ref('stg_census__port_hs6_monthly') }} f
join {{ ref('dim_port') }} d using (port_code)
group by f.month, d.region, d.gateway_type
