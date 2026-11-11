-- One row per Census port of unlading, with state, region and gateway type.
with ports as (
    select
        port_code,
        any_value(port_name)                      as port_name,
        sum(value_usd)                            as total_value_usd,
        sum(air_value_usd)                        as total_air_value_usd
    from {{ ref('stg_census__port_hs6_monthly') }}
    group by port_code
),
parsed as (
    select
        *,
        nullif(trim(regexp_extract(port_name, ',\s*([A-Z]{2})$', 1)), '') as state
    from ports
)
select
    p.port_code,
    p.port_name,
    p.state,
    coalesce(r.region, 'unknown')                 as region,
    coalesce(r.region_ko, '미상')                  as region_ko,
    case
        when p.port_name like '%AIRPORT%' or p.total_air_value_usd >= 0.95 * p.total_value_usd then 'air'
        else 'sea_land'
    end                                           as gateway_type,
    p.total_value_usd
from parsed p
left join {{ ref('state_region') }} r on r.state = p.state
