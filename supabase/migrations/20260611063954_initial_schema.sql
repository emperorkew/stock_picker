CREATE TYPE signal_type AS ENUM ('BUY', 'HOLD', 'SELL');
CREATE TYPE trade_action AS ENUM ('BUY', 'SELL');

CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(12) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE market_snapshots (
    id BIGSERIAL PRIMARY KEY,
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,

    -- Prijs & Volume (Fundamenteel)
    price NUMERIC(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    volume_avg_20d BIGINT,
    pe_ratio NUMERIC(6, 2),
    fcf_yield NUMERIC(5, 2),

    -- Technische Indicatoren
    ema_50 NUMERIC(12, 4),
    ema_200 NUMERIC(12, 4),
    rsi_14 NUMERIC(5, 2),
    macd_line NUMERIC(12, 4),
    macd_signal NUMERIC(12, 4),

    UNIQUE (asset_id, timestamp)
);

CREATE TABLE trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    signal signal_type NOT NULL,
    confidence_score NUMERIC(4, 2), -- Waarde tussen 0.00 en 1.00
    rationale TEXT
);

CREATE TABLE portfolio_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID REFERENCES assets(id) ON DELETE RESTRICT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    action trade_action NOT NULL,
    quantity NUMERIC(12, 6) NOT NULL,
    price_per_unit NUMERIC(12, 4) NOT NULL,
    total_amount NUMERIC(12, 2) GENERATED ALWAYS AS (quantity * price_per_unit) STORED,
    transaction_fee NUMERIC(6, 2) DEFAULT 0.00
);

CREATE INDEX idx_snapshots_asset_timestamp
ON market_snapshots (asset_id, timestamp DESC);

CREATE INDEX idx_signals_asset_timestamp
ON trading_signals (asset_id, timestamp DESC);

-- Keep assets.updated_at current on every UPDATE.
CREATE EXTENSION IF NOT EXISTS moddatetime;

CREATE TRIGGER assets_updated_at
    BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION moddatetime(updated_at);

-- Row Level Security: no policies are defined, so the anon key has no
-- access at all. Scripts using the service-role key bypass RLS.
ALTER TABLE assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_ledger ENABLE ROW LEVEL SECURITY;