-- Korean item names (HSK 10-digit) taken from the archived Korea Customs Service responses.
select hsk10, name_ko, cast(last_seen_month || '-01' as date) as last_seen_month
from read_csv(
    '{{ var("extracted_dir") }}/hsk10_names.csv',
    header = true,
    types = {'hsk10': 'VARCHAR', 'name_ko': 'VARCHAR', 'last_seen_month': 'VARCHAR'}
)
