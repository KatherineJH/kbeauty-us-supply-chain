-- Census HS10 x month US imports from the eight pre-registered peer countries (warehouse/seeds/peer_countries.csv).
select
    cast(month || '-01' as date)                                    as month,
    cty_code,
    cty_name,
    hs10,
    value_usd,
    coalesce(vessel_value_usd, 0)                                   as vessel_value_usd,
    coalesce(air_value_usd, 0)                                      as air_value_usd,
    value_usd - coalesce(vessel_value_usd, 0) - coalesce(air_value_usd, 0) as other_value_usd,
    coalesce(vessel_weight_kg, 0)                                   as vessel_weight_kg,
    coalesce(air_weight_kg, 0)                                      as air_weight_kg
from read_csv(
    '{{ var("extracted_dir") }}/peer_hs10_monthly.csv',
    header = true,
    types = {'month': 'VARCHAR', 'cty_code': 'VARCHAR', 'hs10': 'VARCHAR', 'value_usd': 'BIGINT',
             'vessel_value_usd': 'BIGINT', 'air_value_usd': 'BIGINT', 'vessel_weight_kg': 'BIGINT',
             'air_weight_kg': 'BIGINT'}
)
