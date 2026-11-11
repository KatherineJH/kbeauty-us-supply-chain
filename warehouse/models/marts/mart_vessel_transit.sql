-- Busan -> US filing days for observations accepted for statistics, with destination region.
with r as (
    select *, nullif(trim(regexp_extract(us_port_name, ',\s*([A-Z]{2})$', 1)), '') as state
    from {{ ref('stg_vessel__routes') }}
    where use_for_stats
)
select
    r.observation_id,
    date_trunc('month', r.us_filing_date)                         as filing_month,
    r.busan_departure_date,
    r.us_filing_date,
    r.us_port_name,
    coalesce(sr.region, 'unknown')                                as region,
    coalesce(sr.region_ko, '미상')                                 as region_ko,
    r.departure_to_filing_days,
    r.match_method
from r
left join {{ ref('state_region') }} sr on sr.state = r.state
