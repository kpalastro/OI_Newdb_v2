-- Add exit_timestamp and option_symbol columns to option_return_backtest_trades table
-- Run this script to add the new columns for existing tables

ALTER TABLE option_return_backtest_trades 
ADD COLUMN IF NOT EXISTS exit_timestamp TIMESTAMP;

ALTER TABLE option_return_backtest_trades 
ADD COLUMN IF NOT EXISTS option_symbol TEXT;

-- Create index on exit_timestamp for faster queries
CREATE INDEX IF NOT EXISTS idx_backtest_trades_exit_timestamp 
ON option_return_backtest_trades (exit_timestamp);

-- Create index on option_symbol for faster queries
CREATE INDEX IF NOT EXISTS idx_backtest_trades_option_symbol 
ON option_return_backtest_trades (option_symbol);

-- Verify the columns were added
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'option_return_backtest_trades' 
  AND column_name IN ('exit_timestamp', 'option_symbol')
ORDER BY column_name;
