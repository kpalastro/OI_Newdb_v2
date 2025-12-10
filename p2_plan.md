# Detailed Technical Implementation Plan: Medium Priority (P2) Enhancements

This document outlines the implementation steps for Phase 2 Medium Priority items: Strategy Router Enhancement, Reinforcement Learning (Execution Optimization), Sentiment Integration, Distributed Processing, and Advanced Visualization.

---

## 1. Strategy Router Enhancement (P2-1)

**Objective**: Transition from a model-centric routing (LightGBM vs DL) to a strategy-centric routing system that selects specific option trading strategies based on market conditions and signal horizons.

### 1.1 New Directory Structure: `strategies/`

Create a new directory `OI_Newdb/strategies/` to house specific trading logic.

#### A. Base Strategy Interface (`OI_Newdb/strategies/base_strategy.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TradeRecommendation:
    signal: str  # BUY, SELL, HOLD
    confidence: float
    strategy_name: str
    rationale: str
    suggested_contract: str  # e.g., 'ATM', 'OTM-1', 'ITM+1'
    metadata: Dict[str, Any]

class BaseStrategy(ABC):
    @abstractmethod
    def analyze(self, signal: Dict, features: Dict, market_state: Dict) -> TradeRecommendation:
        """
        Analyze signal and market state to generate specific trade recommendation.
        """
        pass
```

### 1.2 Strategy Implementations

#### B. Gamma Scalping (`OI_Newdb/strategies/gamma_scalping.py`)
*   **Focus**: Intraday, Low Volatility environment.
*   **Logic**:
    *   Identify ATM options with high Gamma.
    *   Entry: When IV is low (IV Rank < 30).
    *   Exit: Delta neutral hedging or fixed profit target per scalp.

#### C. OI Buildup (`OI_Newdb/strategies/oi_buildup.py`)
*   **Focus**: Swing, Directional.
*   **Logic**:
    *   Detect divergent OI patterns (Price Up + OI Up = Long Build).
    *   Confirm with PCR (Put-Call Ratio) trend.
    *   Entry: Breakout with significant OI support.

#### D. Volatility Expansion (`OI_Newdb/strategies/vol_expansion.py`)
*   **Focus**: Pre-event or Squeeze.
*   **Logic**:
    *   Screen for "Squeeze" (Bollinger Band compression).
    *   Entry: Long Straddle/Strangle when Volatility is expected to expand.

#### E. Expiry Pin (`OI_Newdb/strategies/expiry_pin.py`)
*   **Focus**: Expiry Day (0-DTE).
*   **Logic**:
    *   Calculate Max Pain and Gamma Flip levels.
    *   Predict "pinning" probability using `ExpiryTransformer` (from P1).
    *   Entry: Credit spreads away from the pin level.

### 1.3 Advanced Strategy Router (`OI_Newdb/execution/strategy_router.py`)
*   **Refactor**: Rename/Extend `StrategyRouter` to `AdvancedStrategyRouter`.
*   **Routing Logic**:
    ```python
    def route(self, signal: Dict, features: Dict) -> TradeRecommendation:
        horizon = signal.get('horizon', 'intraday')
        regime = features.get('regime', 'neutral')
        
        if horizon == 'expiry':
            return self.expiry_pin.analyze(signal, features, ...)
        elif regime == 'low_vol' and horizon == 'intraday':
            return self.gamma_scalping.analyze(signal, features, ...)
        elif regime == 'trending':
            return self.oi_buildup.analyze(signal, features, ...)
        # ... fallback logic
    ```

---

## 2. Reinforcement Learning Execution (P2-2)

**Objective**: Optimize trade execution (slippage reduction, fill rate) using RL, replacing simple limit chasing. This differs from the P1 ML models which predict *direction*; this predicts *how to execute*.

### 2.1 Enhanced RL Module (`OI_Newdb/models/reinforcement_learning.py`)

Expand the existing module to support Execution logic.

#### A. Execution Environment (`ExecutionEnvironment`)
*   **State Space**:
    *   `bid_ask_spread`: Current spread width.
    *   `book_imbalance`: Bid volume vs Ask volume ratio.
    *   `volatility_5m`: Short-term volatility.
    *   `time_remaining`: Seconds left to fill the order.
*   **Action Space**:
    *   `price_offset`: Continuous [-2 ticks, +2 ticks] relative to Mid-Price.
    *   `aggression`: Discrete (0=Passive/Limit, 1=Aggressive/Market).
*   **Reward Function**:
    *   `implementation_shortfall`: (Benchmark Price - Fill Price) for Buy.
    *   `fill_penalty`: Large negative reward if order is not filled by deadline.

#### B. RL Executor Agent (`RLExecutor`)
*   **Algorithm**: PPO (Proximal Policy Optimization) via Stable Baselines 3.
*   **Methods**:
    *   `decide_placement(order_params, market_state) -> PlacementDetails`
    *   `train(historical_tick_data)`

### 2.2 Integration with Auto Executor (`OI_Newdb/execution/auto_executor.py`)
*   **Modify**: `execute_paper_trade` method.
*   **Integration**:
    ```python
    if self.use_rl_execution and self.rl_executor.is_ready():
        # Get optimal placement from RL
        placement = self.rl_executor.decide_placement(
            symbol=symbol,
            current_price=current_price,
            spread=spread,
            imbalance=order_book_imbalance
        )
        fill_price = self._simulate_fill(placement, current_price)
    else:
        # Fallback to existing Limit Chasing logic
        fill_price = self._limit_chase_logic(...)
    ```

---

## 3. Sentiment Integration (P2-3)

**Objective**: Integrate Institutional Flow (FII/DII) and Market Breadth (Advance/Decline) as core sentiment indicators. This replaces the originally proposed social media sentiment with hard market data to filter trades and gauge broad market participation.

### 3.1 New Module: `OI_Newdb/sentiment_analyzer.py`

Create a unified sentiment analysis module combining live market breadth and institutional data.

#### Class: `SentimentAnalyzer`
*   **Attributes**:
    *   `fii_dii_cache`: Cached FII/DII data (updated daily).
    *   `breadth_cache`: Last fetched A/D data for Nifty 50/100.
*   **Methods**:
    *   `fetch_fii_dii()`: Retrieves latest FII/DII net values (using logic from `fii_dii.py`).
    *   `fetch_market_breadth()`: Retrieves live Advance/Decline and TRIN (using logic from `adsentiment.py`).
    *   `get_sentiment_metrics() -> Dict`: Returns a unified sentiment dictionary.

**Key Metrics to Generate**:
1.  **`ad_ratio_n50`**: Advance/Decline Ratio (Nifty 50).
2.  **`ad_ratio_n100`**: Advance/Decline Ratio (Nifty 100).
3.  **`trin_index`**: Arms Index (Flow-adjusted breadth).
4.  **`institutional_net`**: Net FII + DII Flow (₹ Cr).
5.  **`sentiment_score`**: Composite 0-100 score based on Breadth and Flow.

### 3.2 Integration with Feature Engineering

*   **Modify**: `OI_Newdb/feature_engineering.py`
*   **Action**:
    *   Inject `SentimentAnalyzer` instance into the feature pipeline.
    *   Add columns: `sentiment_ad_ratio`, `sentiment_trin`, `sentiment_fii_net`.
    *   These features become available to `AdvancedStrategyRouter`.

### 3.3 Integration with Strategy Router

*   **Modify**: `OI_Newdb/strategies/base_strategy.py` (and subclasses)
*   **Action**:
    *   Ensure `market_state` passed to `analyze()` includes sentiment metrics.
    *   **Logic Update**:
        *   *Gamma Scalping*: Prefer Positive Breadth for Long Gamma.
        *   *OI Buildup*: Require Breadth confirmation for Swing trades (e.g., "Buy Signal" requires `ad_ratio > 1.0`).

---

## 4. Distributed Processing (P2-4)

**Objective**: Offload heavy feature engineering and ML inference from the main application thread to a dedicated worker process. This ensures the WebSocket consumer (main thread) remains responsive to incoming ticks, preventing lag during high-load periods (e.g., market open).

### 4.1 New Module: `OI_Newdb/distributed/feature_worker.py`

Create a dedicated worker process using Python's `multiprocessing`.

#### Class: `FeatureWorkerProcess`
*   **Inherits**: `multiprocessing.Process`
*   **Attributes**:
    *   `input_queue`: `multiprocessing.Queue` (Receives tick data/snapshots)
    *   `output_queue`: `multiprocessing.Queue` (Sends signals/features back)
    *   `model_registry`: Loads ML models in its own memory space.
*   **Method `run()`**:
    *   Infinite loop popping items from `input_queue`.
    *   Calls `engineer_live_feature_set()` (CPU intensive).
    *   Calls `MLSignalGenerator.generate_signal()` (CPU intensive).
    *   Pushes result `{signal, confidence, features}` to `output_queue`.

### 4.2 Integration with AppManager

*   **Modify**: `OI_Newdb/app_manager.py`
*   **Action**:
    *   Initialize `FeatureWorkerProcess` in `__init__`.
    *   Start the worker in `start_modules()`.
    *   In `_process_tick()`: Instead of computing features inline, push `tick_data` to `input_queue`.
    *   Add a new thread `ResultConsumerThread` to poll `output_queue` and trigger `AutoExecutor`.

---

## 5. Advanced Visualization (P2-5)

**Objective**: Replace the CLI-based viewer with a real-time web dashboard using FastAPI and WebSockets. This provides live charts for Price, OI, IV, and Signals.

### 5.1 New Module: `OI_Newdb/dashboard/server.py`

Create a lightweight web server.

#### Class: `DashboardServer`
*   **Framework**: `FastAPI` with `uvicorn`.
*   **Endpoints**:
    *   `GET /`: Serves `dashboard.html`.
    *   `GET /api/history`: Returns historical 5-minute bars for charting.
    *   `WS /ws/live`: WebSocket endpoint broadcasting updates every second.
*   **State Access**:
    *   Requires reference to `AppManager` (passed via dependency injection or singleton pattern) to read `data_reels` and `open_positions`.

### 5.2 Frontend: `OI_Newdb/templates/dashboard.html`

Enhance the existing template.

*   **Libraries**: `Chart.js` (or `Plotly.js`) for rendering.
*   **Components**:
    *   **Main Chart**: Candlestick (Price) + Line (VWAP).
    *   **Sub Chart 1**: OI Change (Call vs Put) - Bar Chart.
    *   **Sub Chart 2**: IV & VIX - Line Chart.
    *   **Signal Panel**: Live table of ML signals and Confidence.
    *   **Position Panel**: Live PnL and active trades.

### 5.3 Integration

*   **Modify**: `OI_Newdb/oi_tracker_new.py`
*   **Action**:
    *   Launch `DashboardServer` in a daemon thread on startup (default port 8000).
    *   Ensure `AppManager` exposes a thread-safe `get_latest_state()` method for the dashboard to poll.

---

## 6. Execution Schedule

| Week | Component | Deliverables | Files Touched |
| :--- | :--- | :--- | :--- |
| **1** | **Strategies** | Base Class, Gamma Scalping, OI Buildup | `strategies/*.py` |
| **2** | **Strategies** | Vol Expansion, Expiry Pin, Router | `strategies/*.py`, `strategy_router.py` |
| **3** | **RL Core** | Execution Env, PPO Agent | `models/reinforcement_learning.py` |
| **4** | **RL Integration** | AutoExecutor | `auto_executor.py` |
| **5** | **Sentiment** | Sentiment Analyzer | `sentiment_analyzer.py`, `feature_engineering.py` |
| **6** | **Distributed** | Feature Worker Process | `distributed/feature_worker.py`, `app_manager.py` |
| **7** | **Dashboard** | FastAPI Server, HTML UI | `dashboard/server.py`, `templates/dashboard.html` |
