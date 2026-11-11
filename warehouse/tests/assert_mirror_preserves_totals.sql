-- The mirror mart must keep every dollar of both sources in every month the US has published: Korean exports
-- and US imports per month must equal their staging totals, and the published flag must be month-level.
with kr_stg as (select month, sum(export_usd) as v from {{ ref('stg_kcs__kr_exports_us') }} group by month),
     us_stg as (select month, sum(value_usd) as v from {{ ref('stg_census__hs10_monthly') }} group by month),
     m as (select month, sum(kr_export_usd) as kr, sum(us_import_usd) as us,
                  count(distinct us_published) as flag_values
           from {{ ref('mart_mirror_monthly') }} group by month)
select m.month, 'korea total changed' as issue from m join kr_stg k on k.month = m.month where m.kr is distinct from k.v
union all
select m.month, 'us total changed' from m join us_stg u on u.month = m.month where m.us is distinct from u.v
union all
select month, 'published flag differs within a month' from m where flag_values > 1
