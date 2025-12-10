# Detailed Technical Implementation Plan: High Priority (P1) Enhancements

This document provides a low-level technical roadmap for implementing the 5 High Priority recommendations. It includes specific class structures, algorithms, file modifications, and integration points.

---

## 1. Order Flow Analysis (P1-1)
**Objective**: Detect institutional "smart money" activity via block trades and sweep orders.

### 1.1 New Module: `data_ingestion/order_flow_analyzer.py`

#### Class: `OrderFlowAnalyzer`
*   **Attributes**:
    *   `min_block_size`: `int` (default: 1000)
    *   `sweep_window_seconds`: `float` (default: 1.0)
    *   `recent_trades`: `deque` (stores `{timestamp, strike, side, volume}`)
*   **Methods**:
    *   `__init__(self, config: Dict)`
    *   `process_tick(self, tick: Dict) -> Dict`
        *   **Algorithm**:
            1.  Check `tick['volume']`. If `> min_block_size`:
                *   Classify side: `BUY` if `tick['ltp'] >= tick['ask']`, `SELL` if `tick['ltp'] <= tick['bid']`.
                *   Store in internal `block_trades` list.
            2.  Add tick to `recent_trades` deque.
            3.  Prune `recent_trades` older than `sweep_window_seconds`.
            4.  **Sweep Detection**:
                *   Group `recent_trades` by `side`.
                *   If unique `strike` count > 3 AND total volume > `3 * min_block_size`:
                    *   Trigger `sweep_detected` flag.
        *   **Returns**: `{ 'block_count_5m': int, 'sweep_score': float, 'net_flow_imbalance': int }`

### 1.2 Integration Points
1.  **`OI_Newdb/handlers.py`**:
    *   In `ExchangeDataHandler.__init__`: Initialize `self.order_flow = OrderFlowAnalyzer(self.config)`.
    *   In `_update_reels` (or tick processing loop):
        *   `flow_metrics = self.order_flow.process_tick(tick_data)`
        *   Store `flow_metrics` in `self.latest_tick_metadata` or a new `flow_cache`.

2.  **`OI_Newdb/feature_engineering.py`**:
    *   Update `REQUIRED_FEATURE_COLUMNS`: Add `block_trade_count`, `sweep_score`.
    *   In `engineer_live_feature_set`:
        *   Extract metrics from `handler.flow_cache`.
        *   Add to returned feature dict.

---

## 2. Multi-Horizon Models (P1-2)
**Objective**: Specialized models for Intraday (scalping), Swing (1-3 days), and Expiry.

### 2.1 Model Architectures (`OI_Newdb/models/`)

#### A. `intraday_lstm.py` (PyTorch)
*   **Class**: `IntradayLSTM(nn.Module)`
*   **Structure**:
    *   Input: `(Batch, Seq_Len=60, Features=15)` (Past 60 5-sec ticks).
    *   Layer 1: `LSTM(input_size=15, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)`.
    *   Layer 2: `Attention` (Self-attention on LSTM output).
    *   Layer 3: `Linear(64 -> 3)` (Buy, Sell, Hold).
*   **Forward**: `x -> LSTM -> Attention -> GlobalAvgPool -> Linear -> Softmax`.

#### B. `swing_ensemble.py` (Sklearn/XGBoost)
*   **Class**: `SwingEnsemble`
*   **Structure**:
    *   Wraps `XGBClassifier` and `LGBMClassifier`.
    *   Input: Daily aggregated features (OI changes, VIX, PCR).
    *   **Logic**: `predict_proba` returns weighted average of sub-models.

#### C. `horizon_router.py`
*   **Class**: `HorizonRouter`
*   **Method**: `route(context: Dict) -> str`
    *   **Logic**:
        ```python
        if context['days_to_expiry'] < 1/365.0: # Expiry Day
            return 'expiry'
        elif context['strategy_mode'] == 'SCALP':
            return 'intraday'
        else:
            return 'swing'
        ```

### 2.2 Core Integration (`OI_Newdb/ml_core.py`)
*   **Class**: `MultiHorizonEnsemble`
    *   `__init__`: Loads `intraday_state_dict.pt`, `swing_model.pkl`, `expiry_model.pt`.
    *   `predict(features, context)`:
        *   `horizon = self.router.route(context)`
        *   Delegate to specific model.
        *   Return standardized format: `{ 'signal': str, 'confidence': float, 'horizon': str }`.

---

## 3. Expiry-Day Specialized Transformer (P1-3)
**Objective**: Capture non-linear 0-DTE dynamics (Gamma pinning, Max Pain).

### 3.1 New Module: `OI_Newdb/models/expiry_transformer.py`
*   **Class**: `ExpiryTransformer(nn.Module)`
*   **Structure**:
    *   Input: Snapshot of Option Chain (Strikes around ATM).
    *   Embedding: `Linear(feature_dim -> 64)`.
    *   Encoder: `TransformerEncoder(d_model=64, nhead=4, num_layers=3)`.
    *   Heads:
        *   `pin_head`: `Linear(64 -> 1)` (Sigmoid output: Pin Probability).
        *   `direction_head`: `Linear(64 -> 3)` (Softmax).

### 3.2 Feature Engineering (`OI_Newdb/feature_engineering.py`)
*   **New Function**: `calculate_expiry_features(chain)`
    *   **Max Pain**: Iterate all strikes, calc intrinsic value sum for Calls+Puts. Find min.
    *   **Gamma Flip**: Calculate Net Gamma at spot +/- 5%. Find price where sign changes.
    *   **Theta Accel**: (Current Theta - 5min ago Theta) / 5min.

---

## 4. Monte Carlo Simulation (P1-4)
**Objective**: Robustness testing via synthetic path generation.

### 4.1 New Module: `OI_Newdb/backtesting/monte_carlo.py`

#### Class: `MonteCarloTester`
*   **Method**: `run_simulation(backtest_results, n_sims=1000)`
    *   **Algorithm (Block Bootstrap)**:
        1.  Take `daily_returns` from backtest.
        2.  Split into blocks of 5 days (preserve volatility clustering).
        3.  For `i` in `0..n_sims`:
            *   Resample blocks with replacement to match original length.
            *   Construct cumulative equity curve.
            *   Calc Max DD, Sharpe.
        4.  Compute Percentiles: 5th, 50th, 95th for Sharpe & Max DD.
*   **Output**: JSON Report `{ 'var_95': float, 'probability_of_loss': float }`.

### 4.2 Integration
*   Modify `OI_Newdb/backtesting/run.py`:
    *   After `engine.run()`:
        *   `mc = MonteCarloTester()`
        *   `mc_results = mc.run_simulation(engine.results)`
        *   Save combined report.

---

## 5. Market Regime Enhancement (P1-5)
**Objective**: Dynamic capital allocation based on granular market states.

### 5.1 Enhanced Module: `OI_Newdb/regime_analysis.py`

#### Class: `MarketRegimeDetector`
*   **Method**: `detect_regime(data: DataFrame) -> str`
    *   **Metrics**:
        *   `ADX` (Trend Strength): using `ta-lib` or pandas rolling.
        *   `VIX_Slope`: 5-day slope of VIX.
        *   `Breadth`: (Adv - Dec) / Total.
    *   **Rules**:
        *   `if ADX > 25 and MA_20 > MA_50`: **TRENDING_UP**
        *   `if ADX > 25 and MA_20 < MA_50`: **TRENDING_DOWN**
        *   `if VIX > 25 and VIX_Slope > 0`: **HIGH_VOL_CRASH**
        *   `else`: **RANGE_BOUND**

### 5.2 Capital Allocation (`OI_Newdb/risk_manager.py`)
*   **New Method**: `get_regime_scaling(regime: str) -> float`
    *   Map:
        *   `TRENDING_UP` -> 1.0 (Full Size)
        *   `RANGE_BOUND` -> 0.6 (Reduced Size)
        *   `HIGH_VOL_CRASH` -> 0.3 (Capital Preservation)
*   **Update**: `check_trade` logic to multiply `max_position_size` by this scaling factor.

---

## 6. Comprehensive Execution Schedule

| Week | Phase | Deliverables | Files Touched |
| :--- | :--- | :--- | :--- |
| **1** | **Order Flow** | `OrderFlowAnalyzer`, `feature_engineering` updates | `order_flow_analyzer.py`, `handlers.py` |
| **2** | **Multi-Horizon** | `IntradayLSTM`, `SwingEnsemble`, `Router` | `models/*.py`, `ml_core.py` |
| **3** | **Expiry Model** | `ExpiryTransformer`, Max Pain logic | `expiry_transformer.py`, `greeks_calculator.py` |
| **4** | **Regime & Risk** | Enhanced Detector, Risk Scaling | `regime_analysis.py`, `risk_manager.py` |
| **5** | **Monte Carlo** | Simulation Engine, Reporting | `monte_carlo.py`, `backtesting/run.py` |
