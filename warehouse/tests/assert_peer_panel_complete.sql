-- Every peer country must have imports in every month of the Korean series; a missing country-month would
-- silently drop out of the difference-in-differences panel.
with months as (select distinct month from {{ ref('stg_census__hs10_monthly') }}),
     grid as (select p.cty_code, m.month from {{ ref('peer_countries') }} p cross join months m),
     have as (select distinct cty_code, month from {{ ref('stg_census__peer_hs10_monthly') }})
select g.cty_code, g.month
from grid g left join have h on h.cty_code = g.cty_code and h.month = g.month
where h.month is null
