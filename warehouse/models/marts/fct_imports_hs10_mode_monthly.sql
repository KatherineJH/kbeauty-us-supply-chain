-- Long format: month x hs10 x transport mode. Weight is not published for the "other" mode.
with base as (
    select s.*, c.category, c.category_ko
    from {{ ref('stg_census__hs10_monthly') }} s
    join {{ ref('hs10_category') }} c using (hs10)
),
unpivoted as (
    select month, hs10, category, category_ko, 'vessel' as mode, vessel_value_usd as value_usd, vessel_weight_kg as weight_kg from base
    union all
    select month, hs10, category, category_ko, 'air', air_value_usd, air_weight_kg from base
    union all
    select month, hs10, category, category_ko, 'other', other_value_usd, null from base
)
select
    strftime(month, '%Y-%m') || '|' || hs10 || '|' || mode  as grain_key,
    *,
    case when weight_kg > 0 then value_usd / weight_kg end as unit_value_usd_per_kg
from unpivoted
