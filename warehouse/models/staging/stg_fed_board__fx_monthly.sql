-- Federal Reserve Board G.5 monthly average exchange rates (release XML, pipeline/fetch_fed_board_fx.py),
-- normalised to local currency units per US dollar.
select
    cast(month || '-01' as date)                                    as month,
    currency,
    lcu_per_usd
from read_csv(
    '{{ var("extracted_dir") }}/fx_monthly.csv',
    header = true,
    types = {'month': 'VARCHAR', 'currency': 'VARCHAR', 'lcu_per_usd': 'DOUBLE'}
)
