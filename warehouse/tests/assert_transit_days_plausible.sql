-- A Busan -> US ocean leg outside 5-60 days indicates a wrong vessel match.
select observation_id, us_port_name, departure_to_filing_days
from {{ ref('mart_vessel_transit') }}
where departure_to_filing_days not between 5 and 60
