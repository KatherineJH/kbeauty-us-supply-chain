-- Month x product category, all modes. Unit value uses vessel + air only (weight is published only for those).
select
    month,
    category,
    any_value(category_ko)                                      as category_ko,
    sum(value_usd)                                              as value_usd,
    sum(case when mode <> 'other' then value_usd end)           as measured_value_usd,
    sum(weight_kg)                                              as weight_kg,
    sum(case when mode <> 'other' then value_usd end) / nullif(sum(weight_kg), 0) as usd_per_kg
from {{ ref('fct_imports_hs10_mode_monthly') }}
group by month, category
