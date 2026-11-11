-- Korean export series must run without gaps from 2024-01 to its last month and must cover every
-- month of the US import series (the mirror comparison needs both sides for each month).
with m as (select distinct month from {{ ref('stg_kcs__kr_exports_us') }}),
     expected as (
        select cast(unnest(generate_series(
            date '2024-01-01', (select max(month) from m), interval 1 month)) as date) as month
     ),
     us as (select distinct month from {{ ref('stg_census__hs10_monthly') }})
select e.month, 'missing in Korean exports' as issue from expected e left join m on m.month = e.month where m.month is null
union all
select u.month, 'US month not covered' from us u left join m on m.month = u.month where m.month is null
