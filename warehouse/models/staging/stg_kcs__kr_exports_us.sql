-- Korea Customs Service: Korea -> United States trade by Korean HSK 10-digit code and month.
-- Export value is FOB (USD); the US side (Census) reports customs value. Both exclude freight and insurance.
select
    cast(month || '-01' as date)                                    as month,
    hsk10,
    left(hsk10, 4)                                                  as hs4,
    left(hsk10, 6)                                                  as hs6,
    export_usd,
    export_kg,
    import_usd,
    import_kg
from read_csv(
    '{{ var("extracted_dir") }}/kr_exports_us_hs10_monthly.csv',
    header = true,
    types = {'month': 'VARCHAR', 'hsk10': 'VARCHAR', 'export_usd': 'BIGINT', 'export_kg': 'BIGINT',
             'import_usd': 'BIGINT', 'import_kg': 'BIGINT'}
)
