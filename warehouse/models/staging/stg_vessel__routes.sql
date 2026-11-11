-- Busan departure (PORT-MIS) matched to the next US entrance filing (CBP CF-1400).
-- vessel_type is dropped: the source column was saved in a legacy Korean encoding and is unreadable.
select
    observation_id,
    call_sign,
    vessel_name,
    operator_name,
    busan_departure_datetime,
    cast(busan_departure_datetime as date)                          as busan_departure_date,
    us_port_code,
    case when us_port_name = 'NEW YORK/NEWARK AREA' then 'NEW YORK/NEWARK AREA, NJ' else us_port_name end
                                                                    as us_port_name,
    us_filing_date,
    departure_to_filing_days,
    match_method,
    match_grade,
    use_for_stats = 'Y'                                             as use_for_stats
from read_csv(
    '{{ var("vessel_routes_path") }}',
    header = true,
    types = {'us_port_code': 'VARCHAR', 'departure_to_filing_days': 'INTEGER', 'us_filing_date': 'DATE'}
)
