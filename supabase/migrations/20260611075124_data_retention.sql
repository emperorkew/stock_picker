-- Automatic data retention: purge old rows daily so the database doesn't
-- grow unboundedly. portfolio_ledger is intentionally kept forever — it is
-- the accounting record and stays small.

create extension if not exists pg_cron;

-- cron.schedule upserts by job name, so re-running this migration is safe.
select cron.schedule(
    'purge-old-market-snapshots',
    '0 3 * * *',
    $$delete from market_snapshots where timestamp < now() - interval '2 years'$$
);

select cron.schedule(
    'purge-old-trading-signals',
    '15 3 * * *',
    $$delete from trading_signals where timestamp < now() - interval '1 year'$$
);
