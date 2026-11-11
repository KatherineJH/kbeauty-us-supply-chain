-- A mode with value but zero weight would create an infinite unit value.
-- Tiny values (< $5,000) are allowed because Census can publish them without a weight.
select month, hs10, mode, value_usd, weight_kg
from {{ ref('fct_imports_hs10_mode_monthly') }}
where mode <> 'other' and value_usd >= 5000 and coalesce(weight_kg, 0) = 0
