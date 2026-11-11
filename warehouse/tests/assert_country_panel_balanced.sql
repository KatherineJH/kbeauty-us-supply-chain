-- The comparison panel must be balanced: 9 countries x every month, one row each, and the Korean rows must
-- equal the Korea-only summary mart (the two paths read the same Census file).
with counts as (
    select count(*) as n, count(distinct cty_code) as countries, count(distinct month) as months
    from {{ ref('mart_country_panel_monthly') }}
),
kr as (
    select p.month, p.value_usd as panel, s.value_usd as summary
    from {{ ref('mart_country_panel_monthly') }} p
    join {{ ref('mart_monthly_summary') }} s on s.month = p.month
    where p.cty_code = '5800'
)
select 'unbalanced' as issue, cast(n as varchar) as detail from counts where n <> countries * months or countries <> 9
union all
select 'korea mismatch', cast(month as varchar) from kr where panel <> summary
