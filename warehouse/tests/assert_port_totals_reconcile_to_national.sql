-- The port-level file and the national HS10 file come from two Census endpoints. Their monthly totals must
-- agree to the dollar; any difference means a dropped port, HS code or month at extraction.
with n as (select month, sum(value_usd) as v from {{ ref('stg_census__hs10_monthly') }} group by month),
     p as (select month, sum(value_usd) as v from {{ ref('stg_census__port_hs6_monthly') }} group by month)
select coalesce(n.month, p.month) as month, n.v as national, p.v as ports
from n full outer join p on n.month = p.month
where n.v is distinct from p.v
