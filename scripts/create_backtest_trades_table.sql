-- Create option_return_backtest_trades table
-- This table stores detailed backtest trade records with feature payloads and model predictions

CREATE TABLE IF NOT EXISTS option_return_backtest_trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    exchange TEXT NOT NULL,
    option_type TEXT NOT NULL,
    horizon TEXT NOT NULL,
    predicted_return DOUBLE PRECISION,
    actual_return DOUBLE PRECISION,
    predicted_ce_return DOUBLE PRECISION,
    predicted_pe_return DOUBLE PRECISION,
    ce_confidence DOUBLE PRECISION,
    pe_confidence DOUBLE PRECISION,
    turning_point_prob DOUBLE PRECISION,
    recommendation TEXT,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    quantity_lots INTEGER,
    gross_pnl DOUBLE PRECISION,
    net_pnl DOUBLE PRECISION,
    transaction_cost DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    feature_payload JSONB,
    model_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_backtest_trades_timestamp 
ON option_return_backtest_trades (timestamp);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_exchange 
ON option_return_backtest_trades (exchange);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_option_type 
ON option_return_backtest_trades (option_type);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_horizon 
ON option_return_backtest_trades (horizon);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_created_at 
ON option_return_backtest_trades (created_at);

-- Create GIN index for JSONB columns (for efficient JSON queries)
CREATE INDEX IF NOT EXISTS idx_backtest_trades_feature_payload 
ON option_return_backtest_trades USING GIN (feature_payload);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_model_metadata 
ON option_return_backtest_trades USING GIN (model_metadata);

-- Verify the table was created
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'option_return_backtest_trades'
ORDER BY ordinal_position;
