-- Vessel + air value can never exceed the general import value.
select month, hs10, value_usd, vessel_value_usd, air_value_usd
from {{ ref('stg_census__hs10_monthly') }}
where other_value_usd < 0
