-- Census HS10 x month, US total (district "-" rows). One row per month x hs10.
select
    cast(month || '-01' as date)                                    as month,
    hs10,
    hs10_desc,
    value_usd,
    coalesce(vessel_value_usd, 0)                                   as vessel_value_usd,
    coalesce(air_value_usd, 0)                                      as air_value_usd,
    value_usd - coalesce(vessel_value_usd, 0) - coalesce(air_value_usd, 0) as other_value_usd,
    coalesce(vessel_weight_kg, 0)                                   as vessel_weight_kg,
    coalesce(air_weight_kg, 0)                                      as air_weight_kg,
    quantity,
    quantity_unit
from read_csv(
    '{{ var("extracted_dir") }}/hs10_monthly.csv',
    header = true,
    types = {'month': 'VARCHAR', 'hs10': 'VARCHAR', 'value_usd': 'BIGINT', 'vessel_value_usd': 'BIGINT',
             'air_value_usd': 'BIGINT', 'vessel_weight_kg': 'BIGINT', 'air_weight_kg': 'BIGINT', 'quantity': 'BIGINT'}
)
