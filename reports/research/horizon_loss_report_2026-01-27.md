# Horizon Loss Analysis Report

## Data coverage
- Trade logs loaded: **55** trades
- paper_trading_metrics rows with `metadata ? 'horizon'`: **103** rows
- Matched trades ↔ metrics: **11** trades

## Performance by metadata.horizon

| ptm_horizon   |   trades |   win_rate |   total_pnl |   avg_pnl |
|:--------------|---------:|-----------:|------------:|----------:|
| swing         |        6 |   0.333333 |    -1131    |   -188.5  |
| expiry        |        5 |   0.6      |     3077.75 |    615.55 |

## Performance by horizon × exchange × option_type

| ptm_horizon   | exchange   | option_type   |   trades |   win_rate |   total_pnl |   avg_pnl |
|:--------------|:-----------|:--------------|---------:|-----------:|------------:|----------:|
| swing         | BSE        | CE            |        5 |       0.4  |    -1131    |   -226.2  |
| expiry        | NSE        | CE            |        1 |       0    |     -393.25 |   -393.25 |
| swing         | BSE        | PE            |        1 |       0    |        0    |      0    |
| expiry        | NSE        | PE            |        4 |       0.75 |     3471    |    867.75 |

## Loss concentration by exit_reason

| ptm_horizon   | exit_reason                 |   trades |   win_rate |   total_pnl |
|:--------------|:----------------------------|---------:|-----------:|------------:|
| swing         | Stop Loss (-50)             |        3 |        0   |    -3212    |
| expiry        | Signal Flip (BUY → SELL)    |        1 |        0   |     -393.25 |
| expiry        | End of Day Exit (15:20 IST) |        2 |        0.5 |     -113.75 |
| swing         | End of Day Exit (15:20 IST) |        1 |        0   |        0    |
| swing         | Target Hit (+50)            |        2 |        1   |     2081    |
| expiry        | Target Hit (+25)            |        2 |        1   |     3584.75 |

## How to convert losing trades to winners (actionable)

- Loss drivers by exit_reason: Stop Loss (-50) (trades=3, total_pnl=-3212.00); End of Day Exit (15:20 IST) (trades=2, total_pnl=-607.75); Signal Flip (BUY → SELL) (trades=1, total_pnl=-393.25)
- Worst segments (consider blocking or tightening filters): horizon=swing, BSE CE (trades=5, win_rate=0.40, total_pnl=-1131.00)
- Tactical fixes that usually convert stop-loss losers into avoided trades: add a per-exchange cooldown after any stop-loss (e.g. 10–15 minutes), block re-entry on the same symbol for N minutes, and tighten entry filters for the loss-heavy segment (commonly NSE PE).
- If you want to convert losers into winners via exits (not just avoiding entries): use earlier time-based exit (e.g. if not green by +X minutes, exit), and force-exit on first signal flip instead of waiting.
