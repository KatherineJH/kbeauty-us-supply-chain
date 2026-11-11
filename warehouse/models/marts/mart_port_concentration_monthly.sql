-- Concentration of vessel imports across ports (HHI on vessel value shares, 0-10,000 scale).
with v as (
    select month, port_code, sum(vessel_value_usd) as vessel_value_usd
    from {{ ref('stg_census__port_hs6_monthly') }}
    group by month, port_code
    having sum(vessel_value_usd) > 0
),
s as (
    select v.*, d.port_name, d.region,
           vessel_value_usd / sum(vessel_value_usd) over (partition by month) as share
    from v join {{ ref('dim_port') }} d using (port_code)
)
select
    month,
    count(*)                                                      as vessel_ports,
    round(sum(share * share) * 10000, 1)                          as hhi_vessel,
    sum(case when port_code in ('2704', '2709') then share else 0 end) as la_lb_share,
    sum(case when region = 'west_coast' then share else 0 end)    as west_coast_share,
    sum(case when region = 'east_coast' then share else 0 end)    as east_coast_share,
    sum(case when region = 'gulf_coast' then share else 0 end)    as gulf_coast_share,
    arg_max(port_name, share)                                     as top_port
from s
group by month
order by month
