"""
Walk-Forward Testing Framework for Strategy Validation.

Implements multi-period walk-forward testing to evaluate strategy robustness
across different market conditions.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from risk_manager import calculate_trading_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing."""
    exchange: str
    start_date: date
    end_date: date
    train_days: int = 60          # Training window size (days)
    test_days: int = 20            # Test window size (days)
    step_days: int = 10            # Step size between segments (days)
    
    # Backtest configuration (passed to each segment)
    strategy: str = "ml_signal"
    holding_period_minutes: int = 15
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 1.0
    min_confidence: float = 0.55
    account_size: float = 1_000_000.0
    margin_per_lot: float = 75_000.0
    max_risk_per_trade: float = 0.02
    
    # Model retraining options
    retrain_each_segment: bool = True  # Retrain model for each segment
    model_params: Optional[Dict] = None  # Fixed model params (if not retraining)
    
    def __post_init__(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        if self.train_days <= 0 or self.test_days <= 0 or self.step_days <= 0:
            raise ValueError("train_days, test_days, and step_days must be positive")
        if self.step_days > self.test_days:
            LOGGER.warning("step_days > test_days may cause overlapping test periods")


@dataclass
class SegmentResult:
    """Results for a single walk-forward segment."""
    segment_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    
    # Test period results
    num_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    
    # Model performance
    model_accuracy: Optional[float] = None
    model_f1_score: Optional[float] = None
    
    # Market regime indicators
    avg_volatility: Optional[float] = None
    avg_vix: Optional[float] = None
    market_trend: Optional[str] = None  # 'bull', 'bear', 'neutral'
    
    metadata: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward test results."""
    config: WalkForwardConfig
    segments: List[SegmentResult]
    
    # Aggregated metrics
    avg_sharpe: float
    std_sharpe: float
    avg_max_drawdown: float
    avg_win_rate: float
    avg_profit_factor: float
    
    # Best/worst segments
    best_segment_id: int
    worst_segment_id: int
    
    # Consistency metrics
    positive_periods: int  # Number of periods with positive returns
    consistency_ratio: float  # Ratio of positive periods to total
    
    # Statistical analysis
    sharpe_distribution: Dict[str, float]  # min, 25th, median, 75th, max
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': {
                'exchange': self.config.exchange,
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'train_days': self.config.train_days,
                'test_days': self.config.test_days,
                'step_days': self.config.step_days
            },
            'summary': {
                'num_segments': len(self.segments),
                'avg_sharpe': self.avg_sharpe,
                'std_sharpe': self.std_sharpe,
                'avg_max_drawdown': self.avg_max_drawdown,
                'avg_win_rate': self.avg_win_rate,
                'avg_profit_factor': self.avg_profit_factor,
                'consistency_ratio': self.consistency_ratio,
                'positive_periods': self.positive_periods
            },
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'train_start': seg.train_start.isoformat(),
                    'train_end': seg.train_end.isoformat(),
                    'test_start': seg.test_start.isoformat(),
                    'test_end': seg.test_end.isoformat(),
                    'num_trades': seg.num_trades,
                    'total_pnl': seg.total_pnl,
                    'sharpe_ratio': seg.sharpe_ratio,
                    'max_drawdown': seg.max_drawdown,
                    'win_rate': seg.win_rate,
                    'profit_factor': seg.profit_factor
                }
                for seg in self.segments
            ]
        }


class WalkForwardTester:
    """
    Multi-period walk-forward testing framework.
    
    Splits historical data into train/test segments and evaluates strategy
    performance across all segments to assess robustness.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.segments: List[Dict] = []
    
    def generate_segments(self) -> List[Dict]:
        """
        Generate train/test segment windows.
        
        Returns:
            List of segment dicts with 'train_start', 'train_end', 'test_start', 'test_end'
        """
        segments = []
        current_start = self.config.start_date
        segment_id = 0
        
        while True:
            train_end = current_start + timedelta(days=self.config.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_days)
            
            # Stop if test period exceeds end date
            if test_end > self.config.end_date:
                break
            
            segments.append({
                'segment_id': segment_id,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            segment_id += 1
            current_start += timedelta(days=self.config.step_days)
        
        self.segments = segments
        LOGGER.info(f"Generated {len(segments)} walk-forward segments")
        return segments
    
    def run(self) -> WalkForwardResult:
        """
        Run walk-forward test across all segments.
        
        Returns:
            WalkForwardResult with aggregated metrics
        """
        if not self.segments:
            self.generate_segments()
        
        if not self.segments:
            LOGGER.warning("No segments generated for walk-forward test")
            return self._empty_result()
        
        segment_results: List[SegmentResult] = []
        
        for segment in self.segments:
            LOGGER.info(
                f"Processing segment {segment['segment_id']}: "
                f"Train {segment['train_start']} to {segment['train_end']}, "
                f"Test {segment['test_start']} to {segment['test_end']}"
            )
            
            # Train model on training period (if enabled)
            if self.config.retrain_each_segment:
                self._train_model_for_segment(
                    segment['train_start'],
                    segment['train_end']
                )
            
            # Run backtest on test period
            segment_result = self._run_segment_backtest(segment)
            segment_results.append(segment_result)
        
        # Aggregate results
        return self._aggregate_results(segment_results)
    
    def _train_model_for_segment(self, train_start: date, train_end: date):
        """
        Train model on training segment.
        
        This would integrate with train_orchestrator or ml_core to retrain models.
        For now, this is a placeholder - actual implementation depends on model training pipeline.
        """
        # TODO: Integrate with model training pipeline
        # Options:
        # 1. Call train_orchestrator.train_model() with train_start/train_end
        # 2. Use MLSignalGenerator's retrain capability (if available)
        # 3. Load pre-trained models for each segment
        
        LOGGER.debug(f"Training model for period {train_start} to {train_end}")
        # Placeholder - actual training logic goes here
        # For now, we'll use existing models (no retraining)
        pass
    
    def _run_segment_backtest(self, segment: Dict) -> SegmentResult:
        """
        Run backtest for a single segment.
        
        Args:
            segment: Segment dict with train/test dates
        
        Returns:
            SegmentResult with test period metrics
        """
        # Create backtest config for test period
        backtest_config = BacktestConfig(
            exchange=self.config.exchange,
            start=segment['test_start'],
            end=segment['test_end'],
            strategy=self.config.strategy,
            holding_period_minutes=self.config.holding_period_minutes,
            transaction_cost_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
            min_confidence=self.config.min_confidence,
            account_size=self.config.account_size,
            margin_per_lot=self.config.margin_per_lot,
            max_risk_per_trade=self.config.max_risk_per_trade
        )
        
        # Run backtest
        try:
            engine = BacktestEngine(backtest_config)
            backtest_result = engine.run()
        except Exception as e:
            LOGGER.error(f"Error running backtest for segment {segment['segment_id']}: {e}", exc_info=True)
            # Return empty result for this segment
            return SegmentResult(
                segment_id=segment['segment_id'],
                train_start=segment['train_start'],
                train_end=segment['train_end'],
                test_start=segment['test_start'],
                test_end=segment['test_end'],
                num_trades=0,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0
            )
        
        # Extract metrics
        metrics = backtest_result.metrics
        
        # Calculate market regime indicators (simplified)
        avg_volatility, avg_vix, market_trend = self._analyze_market_regime(
            segment['test_start'],
            segment['test_end']
        )
        
        # Extract trading metrics from backtest result
        num_trades = len(backtest_result.trades)
        total_pnl = metrics.get('net_total_pnl', 0.0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        if np.isnan(sharpe_ratio):
            sharpe_ratio = 0.0
        max_drawdown = metrics.get('net_max_drawdown', 0.0)
        win_rate = metrics.get('win_rate', 0.0)
        profit_factor = metrics.get('profit_factor', 0.0)
        if np.isnan(profit_factor):
            profit_factor = 0.0
        avg_trade_pnl = metrics.get('avg_trade_pnl', 0.0)
        
        return SegmentResult(
            segment_id=segment['segment_id'],
            train_start=segment['train_start'],
            train_end=segment['train_end'],
            test_start=segment['test_start'],
            test_end=segment['test_end'],
            num_trades=num_trades,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_volatility=avg_volatility,
            avg_vix=avg_vix,
            market_trend=market_trend,
            metadata={
                'gross_pnl': metrics.get('gross_total_pnl', 0.0),
                'cost_per_trade': metrics.get('cost_per_trade', 0.0),
                'num_trades': num_trades
            }
        )
    
    def _analyze_market_regime(
        self,
        start_date: date,
        end_date: date
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Analyze market regime for test period.
        
        Returns:
            Tuple of (avg_volatility, avg_vix, market_trend)
        """
        # TODO: Load historical data and calculate:
        # - Average realized volatility
        # - Average VIX
        # - Market trend (bull/bear/neutral based on returns)
        
        # Placeholder implementation - would need to query database
        # For now, return None values
        try:
            import database_new as db
            data = db.load_historical_data_for_ml(self.config.exchange, start_date, end_date)
            if not data.empty and 'vix' in data.columns:
                avg_vix = float(data['vix'].mean()) if 'vix' in data.columns else None
                # Calculate realized volatility from price changes
                if 'underlying_price' in data.columns:
                    returns = data['underlying_price'].pct_change().dropna()
                    avg_volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else None
                    # Determine trend
                    total_return = float((data['underlying_price'].iloc[-1] / data['underlying_price'].iloc[0] - 1) * 100) if len(data) > 0 else 0.0
                    if total_return > 2.0:
                        market_trend = 'bull'
                    elif total_return < -2.0:
                        market_trend = 'bear'
                    else:
                        market_trend = 'neutral'
                else:
                    avg_volatility = None
                    market_trend = None
            else:
                avg_volatility = None
                avg_vix = None
                market_trend = None
        except Exception as e:
            LOGGER.debug(f"Could not analyze market regime: {e}")
            avg_volatility = None
            avg_vix = None
            market_trend = None
        
        return avg_volatility, avg_vix, market_trend
    
    def _aggregate_results(self, segment_results: List[SegmentResult]) -> WalkForwardResult:
        """
        Aggregate results across all segments.
        
        Args:
            segment_results: List of SegmentResult objects
        
        Returns:
            WalkForwardResult with aggregated metrics
        """
        if not segment_results:
            return self._empty_result()
        
        # Extract metrics arrays
        sharpe_ratios = [r.sharpe_ratio for r in segment_results]
        max_drawdowns = [r.max_drawdown for r in segment_results]
        win_rates = [r.win_rate for r in segment_results]
        profit_factors = [r.profit_factor for r in segment_results if not np.isnan(r.profit_factor) and r.profit_factor > 0]
        total_pnls = [r.total_pnl for r in segment_results]
        
        # Calculate statistics
        avg_sharpe = float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0
        std_sharpe = float(np.std(sharpe_ratios)) if sharpe_ratios else 0.0
        avg_max_drawdown = float(np.mean(max_drawdowns)) if max_drawdowns else 0.0
        avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
        avg_profit_factor = float(np.mean(profit_factors)) if profit_factors else 0.0
        
        # Find best/worst segments (by Sharpe ratio)
        if sharpe_ratios:
            best_idx = int(np.argmax(sharpe_ratios))
            worst_idx = int(np.argmin(sharpe_ratios))
        else:
            best_idx = 0
            worst_idx = 0
        
        # Consistency metrics
        positive_periods = sum(1 for pnl in total_pnls if pnl > 0)
        consistency_ratio = positive_periods / len(segment_results) if segment_results else 0.0
        
        # Sharpe distribution
        if sharpe_ratios:
            sharpe_sorted = sorted(sharpe_ratios)
            sharpe_distribution = {
                'min': float(sharpe_sorted[0]),
                '25th': float(np.percentile(sharpe_ratios, 25)),
                'median': float(np.median(sharpe_ratios)),
                '75th': float(np.percentile(sharpe_ratios, 75)),
                'max': float(sharpe_sorted[-1])
            }
        else:
            sharpe_distribution = {'min': 0.0, '25th': 0.0, 'median': 0.0, '75th': 0.0, 'max': 0.0}
        
        return WalkForwardResult(
            config=self.config,
            segments=segment_results,
            avg_sharpe=avg_sharpe,
            std_sharpe=std_sharpe,
            avg_max_drawdown=avg_max_drawdown,
            avg_win_rate=avg_win_rate,
            avg_profit_factor=avg_profit_factor,
            best_segment_id=segment_results[best_idx].segment_id if segment_results else 0,
            worst_segment_id=segment_results[worst_idx].segment_id if segment_results else 0,
            positive_periods=positive_periods,
            consistency_ratio=consistency_ratio,
            sharpe_distribution=sharpe_distribution
        )
    
    def _empty_result(self) -> WalkForwardResult:
        """Return empty result when no segments available."""
        return WalkForwardResult(
            config=self.config,
            segments=[],
            avg_sharpe=0.0,
            std_sharpe=0.0,
            avg_max_drawdown=0.0,
            avg_win_rate=0.0,
            avg_profit_factor=0.0,
            best_segment_id=0,
            worst_segment_id=0,
            positive_periods=0,
            consistency_ratio=0.0,
            sharpe_distribution={'min': 0.0, '25th': 0.0, 'median': 0.0, '75th': 0.0, 'max': 0.0}
        )

