-- Census port x HS6 x month (zero-value and all-port total rows were dropped at extraction).
select
    cast(month || '-01' as date)                                    as month,
    port_code,
    coalesce(port_name, 'UNKNOWN ' || port_code)                    as port_name,
    hs6,
    hs6_desc,
    value_usd,
    coalesce(vessel_value_usd, 0)                                   as vessel_value_usd,
    coalesce(air_value_usd, 0)                                      as air_value_usd,
    value_usd - coalesce(vessel_value_usd, 0) - coalesce(air_value_usd, 0) as other_value_usd,
    coalesce(vessel_weight_kg, 0)                                   as vessel_weight_kg,
    coalesce(air_weight_kg, 0)                                      as air_weight_kg,
    coalesce(container_value_usd, 0)                                as container_value_usd,
    coalesce(container_weight_kg, 0)                                as container_weight_kg
from read_csv(
    '{{ var("extracted_dir") }}/hs6_port_monthly.csv',
    header = true,
    types = {'month': 'VARCHAR', 'port_code': 'VARCHAR', 'hs6': 'VARCHAR', 'value_usd': 'BIGINT',
             'vessel_value_usd': 'BIGINT', 'air_value_usd': 'BIGINT', 'vessel_weight_kg': 'BIGINT',
             'air_weight_kg': 'BIGINT', 'container_value_usd': 'BIGINT', 'container_weight_kg': 'BIGINT'}
)
