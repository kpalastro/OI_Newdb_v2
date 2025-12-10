# Daily Operations Manual

## Overview

This guide covers day-to-day operation of the OI ML Trading System, including pre-market setup, market hours monitoring, and post-market tasks.

---

## 1. System Startup

### Pre-Market Checklist (Before 9:00 AM IST)

| # | Task | Command/Action | Verification |
|---|------|----------------|--------------|
| 1 | Activate virtual environment | `source venv_newdb/bin/activate` (Linux) | Terminal shows `(venv_newdb)` |
| 2 | Start the application | `python oi_tracker_new.py` | Shows "Web interface: http://0.0.0.0:5050" |
| 3 | Open web UI | Navigate to `http://localhost:5050` | Login page displays |
| 4 | Authenticate | Enter Zerodha credentials + TOTP | Dashboard loads |
| 5 | Wait for data | System shows "Awaiting tick data" → "Live" | Real-time prices appear |

### Startup Logs to Watch For

```
✓ Expected (Normal):
- "Loaded HMM model with regime mapping" or "Using fallback logic"
- "LightGBM models loaded for NSE"
- "Emitting data_update: underlying_price=XXXX"

⚠ Warnings (Acceptable):
- "HMM model not found... Using fallback logic" → Uses rules-based regime

✗ Errors (Action Required):
- "FATAL ERROR" → Check error message, fix configuration
- "Database connection failed" → Check PostgreSQL is running
```

---

## 2. Market Hours Operation (9:15 AM - 3:30 PM)

### Monitoring Dashboard

The web UI provides real-time monitoring of:

| Panel | What to Monitor | Normal Range |
|-------|----------------|--------------|
| **Underlying Price** | NIFTY/SENSEX spot price | Should update every second |
| **VIX** | India VIX | 10-20 (Normal), >25 (High Vol) |
| **ML Signal** | BUY/SELL/HOLD | Changes based on features |
| **Confidence** | Signal confidence % | >70% for execution |
| **Regime** | Market regime detected | 5 possible regimes |
| **Open Positions** | Paper trading positions | Max 2-3 concurrent |

### Key Metrics to Track

1. **Signal Confidence**: Only trades with >70% confidence are executed
2. **Kelly Fraction**: Risk-adjusted position sizing (>0.15 for execution)
3. **Regime**: Current market condition affecting strategy selection

### What Happens Automatically

1. **ML Feature Engineering**: Every tick → features calculated
2. **Signal Generation**: Features → LightGBM model → BUY/SELL/HOLD
3. **Risk Check**: Signal passes through Advanced Risk Manager
4. **Paper Trade Execution**: If all checks pass → position opened
5. **Position Monitoring**: MTM updated in real-time

---

## 3. Paper Trading - How It Works

### Order Generation Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│ Feature Vector  │───►│ Strategy Router  │───►│  Auto Executor │
│ (70+ features)  │    │ (Model Selection)│    │ (Risk Checks)  │
└─────────────────┘    └──────────────────┘    └────────────────┘
                                                      │
                       ┌──────────────────────────────┘
                       ▼
              ┌─────────────────────────────────────────────┐
              │              PAPER TRADE EXECUTED           │
              │  • Position ID: NSE_AUTO_0001              │
              │  • Strategy: gamma_scalping                │
              │  • Entry: Limit Chase @ LTP + slippage     │
              └─────────────────────────────────────────────┘
```

### Yes - Paper Trading is Fully Automatic!

Once the system is running:
- **No manual order entry required**
- System automatically places paper trades when:
  - Signal = BUY or SELL (not HOLD)
  - Confidence ≥ 70%
  - Kelly Fraction ≥ 0.15
  - Position limits not exceeded
  - Risk checks pass

### Strategy Identification in Orders

Each paper trade includes a **`strategy_name`** field identifying which strategy triggered it:

| Strategy | Trigger Condition | Typical Regime |
|----------|-------------------|----------------|
| `gamma_scalping` | Low VIX + Intraday horizon | LOW_VOL_COMPRESSION |
| `oi_buildup` | OI change + Swing horizon | TRENDING_UP/DOWN |
| `vol_expansion` | Bollinger squeeze detected | Any (breakout setup) |
| `expiry_pin` | Expiry day + 0-DTE horizon | Near expiry |
| `ML_Base` | Default (no specific strategy) | Any |

### Where to See Strategy Info

In the Web UI **Positions** panel:
```
Position ID: NSE_AUTO_0001
Symbol: NIFTY25DEC26500CE
Strategy: gamma_scalping  ← Strategy that triggered trade
Entry Reason: Auto-lightgbm-LimitChase
Confidence: 78.5%
```

---

## 4. Position Management

### Auto Exit Conditions

| Condition | Behavior |
|-----------|----------|
| **Signal Flip** | If BUY position open and new signal is SELL → close position |
| **EOD Exit** | All positions closed at 3:20 PM automatically |
| **Drawdown Stop** | If session loss exceeds configured limit |

### Manual Intervention (If Needed)

Via Web UI:
1. Navigate to **Positions** tab
2. Select position to close
3. Click **Close Position** button

---

## 5. Post-Market Tasks (After 3:30 PM)

### Daily Checklist

| # | Task | Purpose |
|---|------|---------|
| 1 | Review trade log | Check all trades in `trade_logs/` folder |
| 2 | Check metrics | View performance in web UI metrics panel |
| 3 | Export positions | Download daily position summary |
| 4 | Stop application | `Ctrl+C` in terminal |

### Trade Log Location

```
trade_logs/
├── 2025-12-07_NSE_trades.csv
├── 2025-12-07_BSE_trades.csv
└── ...
```

---

## 6. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No signals generated | Check if VIX data is streaming |
| All signals are HOLD | Normal if market is ranging/uncertain |
| Position not executing | Check confidence threshold (needs ≥70%) |
| Database connection error | Ensure PostgreSQL is running |
| WebSocket disconnection | System auto-reconnects; wait 30 seconds |

---

## 7. Configuration Quick Reference

Edit `config.py` or set environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `min_confidence_for_trade` | 0.60 | Minimum confidence to execute |
| `auto_exec_enabled` | True | Enable/disable auto trading |
| `auto_exec_max_position_size_lots` | 4 | Maximum lots per trade |
| `auto_exec_max_open_positions` | 2 | Maximum concurrent positions |
| `auto_exec_close_all_positions_eod` | True | Close all at 3:20 PM |
