-- No missing month between the first and last month (gaps would break YoY and forecasting).
with m as (select distinct month from {{ ref('stg_census__hs10_monthly') }}),
     expected as (
        select cast(unnest(generate_series(
            (select min(month) from m), (select max(month) from m), interval 1 month)) as date) as month
     )
select e.month from expected e left join m on m.month = e.month where m.month is null
