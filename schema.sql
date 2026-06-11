-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_cron";

-- Create Enums
CREATE TYPE signal_type AS ENUM ('BUY', 'HOLD', 'SELL');
CREATE TYPE trade_action AS ENUM ('BUY', 'SELL');

-- Table: assets
CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE
);

-- Table: market_snapshots
CREATE TABLE IF NOT EXISTS market_snapshots (
    id BIGSERIAL PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    price NUMERIC,
    volume NUMERIC,
    volume_avg_20d NUMERIC,
    pe_ratio NUMERIC,
    fcf_yield NUMERIC,
    ema_50 NUMERIC,
    ema_200 NUMERIC,
    rsi_14 NUMERIC,
    macd_line NUMERIC,
    macd_signal NUMERIC
);

-- Table: trading_signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    signal signal_type NOT NULL,
    confidence_score NUMERIC,
    rationale TEXT,
    predicted_return NUMERIC,
    forecast_horizon_minutes INT
);

-- Table: portfolio_ledger
CREATE TABLE IF NOT EXISTS portfolio_ledger (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE RESTRICT,
    timestamp TIMESTAMPTZ NOT NULL,
    action trade_action NOT NULL,
    quantity NUMERIC NOT NULL,
    price_per_unit NUMERIC NOT NULL,
    total_amount NUMERIC GENERATED ALWAYS AS (quantity * price_per_unit) STORED,
    transaction_fee NUMERIC DEFAULT 0
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_market_snapshots_asset_timestamp
    ON market_snapshots(asset_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trading_signals_asset_timestamp
    ON trading_signals(asset_id, timestamp DESC);

-- Automated Retention Policy via pg_cron
-- Schedule a job to delete market_snapshots older than 30 days every midnight
SELECT cron.schedule(
    'prune_market_snapshots',
    '0 0 * * *',
    $$ DELETE FROM market_snapshots WHERE timestamp < NOW() - INTERVAL '30 days'; $$
);