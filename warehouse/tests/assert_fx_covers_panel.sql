-- Each country's currency needs a rate for every month of the import panel (FX is a control variable).
with months as (select distinct month from {{ ref('stg_census__hs10_monthly') }}),
     ccy as (select distinct currency from {{ ref('country_currency') }}),
     grid as (select c.currency, m.month from ccy c cross join months m)
select g.currency, g.month
from grid g left join {{ ref('stg_fed_board__fx_monthly') }} f on f.currency = g.currency and f.month = g.month
where f.lcu_per_usd is null
