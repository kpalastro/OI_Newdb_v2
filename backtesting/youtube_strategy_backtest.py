"""
Backtest script specifically for YouTube Strategy.

This script backtests the YouTube Strategy with its specific rules:
- No trades before 11:00 AM
- OI and PCR alignment required
- VWAP alignment required
- Risk: 30-40 points target, 20-25 points stop
- ATM ± 2-3 strikes only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# Use relative import when run as module, absolute when run as script
try:
    from .engine import BacktestConfig, BacktestResult, TradeRecord
    # Don't import BacktestEngine to avoid loading ML models
except ImportError:
    from backtesting.engine import BacktestConfig, BacktestResult, TradeRecord
# Import YouTube strategy directly - don't need router for backtest
from strategies.youtube_strategy import YouTubeStrategy
from feature_engineering import REQUIRED_FEATURE_COLUMNS
import database_new as db
import numpy as np
from time_utils import now_ist, to_ist


LOGGER = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN and Infinity values with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    else:
        return obj


def _parse_date(value: str):
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def _is_after_11am(timestamp_str: str) -> bool:
    """Check if timestamp is after 11:00 AM IST."""
    try:
        if isinstance(timestamp_str, str):
            # Parse timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            dt_ist = to_ist(dt)
            return dt_ist.time() >= time(11, 0)
        return False
    except Exception as e:
        LOGGER.debug(f"Error parsing timestamp {timestamp_str}: {e}")
        return False


class YouTubeStrategyBacktestEngine:
    """
    Specialized backtest engine for YouTube Strategy.
    
    This engine:
    1. Uses the strategy router with YouTube strategy
    2. Filters trades before 11:00 AM
    3. Uses YouTube strategy's risk parameters
    4. Tracks YouTube-specific metrics
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        # Use YouTube strategy directly (no router needed)
        self.youtube_strategy = YouTubeStrategy()
        
        # YouTube strategy risk parameters
        self.target_points = 35.0  # Average of 30-40
        self.stop_loss_points = 22.5  # Average of 20-25
        
    def run(self) -> BacktestResult:
        """Run backtest with YouTube strategy."""
        frame = self._prepare_frame()
        if frame.empty:
            LOGGER.warning("No data available for %s between %s and %s", 
                          self.config.exchange, self.config.start, self.config.end)
            return BacktestResult(self.config, [], {}, [], raw_rows=0)
        
        LOGGER.warning("Starting YouTube Strategy backtest: %d rows", len(frame))
        
        trades: List[TradeRecord] = []
        predictions: List[int] = []
        actual_returns: List[float] = []
        position_sizes: List[float] = []
        equity_curve: List[Dict[str, float]] = []
        gross_equity = 0.0
        net_equity = 0.0
        net_pnls: List[float] = []
        
        trade_limit = self.config.max_trades or float("inf")
        total_cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000.0
        
        # YouTube strategy specific counters
        before_11am_count = 0
        youtube_hold_count = 0
        oi_pcr_mismatch_count = 0
        vwap_far_count = 0
        low_confidence_count = 0
        zero_capital_count = 0
        signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        strategy_hold_reasons = {}  # Track why strategy returned HOLD
        
        # Store frame reference for look-ahead calculations
        # Use reset_index to ensure integer indices for iloc access
        # This allows us to use iloc for look-ahead without index issues
        frame_reset = frame.reset_index(drop=True)
        
        try:
            # Iterate with enumerate to get index for look-ahead
            for current_idx, (_, row) in enumerate(frame_reset.iterrows()):
                if len(trades) >= trade_limit:
                    break
                
                # Extract timestamp
                try:
                    timestamp_str = str(row['timestamp']) if 'timestamp' in row.index else str(row.get('timestamp', ''))
                except (KeyError, AttributeError):
                    timestamp_str = str(row.get('timestamp', ''))
                
                # Rule 1: No trades before 11:00 AM
                if not _is_after_11am(timestamp_str):
                    before_11am_count += 1
                    continue
                
                # Prepare features
                # Use proper pandas Series access - row is a Series from iterrows()
                features = {}
                for col in REQUIRED_FEATURE_COLUMNS:
                    try:
                        if col in row.index:
                            val = row[col]
                            # Handle Series/DataFrame (shouldn't happen, but be safe)
                            if isinstance(val, pd.Series):
                                val = val.iloc[0] if len(val) > 0 else 0.0
                            elif isinstance(val, pd.DataFrame):
                                val = val.iloc[0, 0] if val.shape[0] > 0 and val.shape[1] > 0 else 0.0
                            
                            # Handle NaN values
                            if pd.isna(val):
                                val = 0.0
                            features[col] = float(val)
                        else:
                            features[col] = 0.0
                    except (KeyError, ValueError, TypeError, IndexError) as e:
                        features[col] = 0.0
                
                # Add flag to force YouTube strategy
                features['use_youtube_strategy'] = True
                
                # Ensure we have all required features with proper defaults
                # Check actual values from row, not just from REQUIRED_FEATURE_COLUMNS
                if 'nse_next_oi_change_call_total' in row.index:
                    try:
                        val = row['nse_next_oi_change_call_total']
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else 0.0
                        if not pd.isna(val):
                            features['nse_next_oi_change_call_total'] = float(val)
                    except:
                        pass
                
                if 'nse_next_oi_change_put_total' in row.index:
                    try:
                        val = row['nse_next_oi_change_put_total']
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else 0.0
                        if not pd.isna(val):
                            features['nse_next_oi_change_put_total'] = float(val)
                    except:
                        pass
                
                # Ensure underlying_price is available
                if 'underlying_price' in row.index:
                    try:
                        val = row['underlying_price']
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else 0.0
                        if not pd.isna(val) and float(val) > 0:
                            features['underlying_price'] = float(val)
                    except:
                        pass
                
                # Ensure PCR features are available
                if 'pcr_total_volume' in row.index:
                    try:
                        val = row['pcr_total_volume']
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else 0.0
                        if not pd.isna(val):
                            features['pcr_total_volume'] = float(val)
                    except:
                        pass
                
                if 'pcr_total_oi' in row.index:
                    try:
                        val = row['pcr_total_oi']
                        if isinstance(val, pd.Series):
                            val = val.iloc[0] if len(val) > 0 else 0.0
                        if not pd.isna(val):
                            features['pcr_total_oi'] = float(val)
                    except:
                        pass
                
                # Create market state for strategy
                market_state = {
                    'strike_difference': 50.0,  # Default for Nifty
                    'handler': None,  # Not available in backtest
                    'exchange': self.config.exchange  # Pass exchange for BSE-specific filters
                }
                
                # Call YouTube strategy directly (no router needed)
                # YouTube strategy doesn't rely on ML signal, it uses OI/PCR/VWAP
                # Create a neutral signal dict - YouTube strategy will determine direction from OI/PCR
                signal_dict = {
                    'signal': 'HOLD',  # YouTube strategy ignores this and uses OI/PCR
                    'confidence': 0.5,
                    'source': 'youtube_direct'
                }
                
                # Use YouTube strategy directly with proper market_state
                youtube_recommendation = self.youtube_strategy.analyze(signal_dict, features, market_state)
                
                # Debug: Log first few to see actual feature values
                if youtube_hold_count < 3:
                    LOGGER.warning("Feature values check:")
                    LOGGER.warning(f"  nse_next_oi_change_call_total: {features.get('nse_next_oi_change_call_total', 'N/A')}")
                    LOGGER.warning(f"  nse_next_oi_change_put_total: {features.get('nse_next_oi_change_put_total', 'N/A')}")
                    LOGGER.warning(f"  pcr_total_volume: {features.get('pcr_total_volume', 'N/A')}")
                    LOGGER.warning(f"  pcr_total_oi: {features.get('pcr_total_oi', 'N/A')}")
                    LOGGER.warning(f"  underlying_price: {features.get('underlying_price', 'N/A')}")
                    LOGGER.warning(f"  time_to_expiry_hours: {features.get('time_to_expiry_hours', 'N/A')}")
                
                # Debug: Log first few HOLD signals to understand why
                if youtube_recommendation.signal == 'HOLD' and youtube_hold_count < 5:
                    LOGGER.debug("HOLD signal #%d: %s", youtube_hold_count + 1, youtube_recommendation.rationale)
                    LOGGER.debug("  Strategy: %s", youtube_recommendation.strategy_name)
                    LOGGER.debug("  OI change call: %s, put: %s", 
                               features.get('nse_next_oi_change_call_total', 'N/A'),
                               features.get('nse_next_oi_change_put_total', 'N/A'))
                    LOGGER.debug("  PCR: %s", features.get('pcr_total_volume', features.get('pcr_total_oi', 'N/A')))
                
                # Check if YouTube strategy returned HOLD
                if youtube_recommendation.signal == 'HOLD':
                    youtube_hold_count += 1
                    rationale = youtube_recommendation.rationale or ''
                    strategy_name = youtube_recommendation.strategy_name or 'unknown'
                    
                    # Log first few to understand pattern
                    if youtube_hold_count <= 5:
                        LOGGER.warning("HOLD #%d - Strategy: %s", youtube_hold_count, strategy_name)
                        LOGGER.warning("  Rationale: %s", rationale[:200] if rationale else 'No rationale')
                        LOGGER.warning("  OI call change: %s, OI put change: %s", 
                                   features.get('nse_next_oi_change_call_total', 'N/A'),
                                   features.get('nse_next_oi_change_put_total', 'N/A'))
                        LOGGER.warning("  PCR volume: %s, PCR OI: %s", 
                                   features.get('pcr_total_volume', 'N/A'),
                                   features.get('pcr_total_oi', 'N/A'))
                        LOGGER.warning("  Underlying price: %s", features.get('underlying_price', 'N/A'))
                    
                    # Track reasons for HOLD
                    rationale_lower = rationale.lower()
                    
                    # Check strategy name first - if not YouTubeStrategy, that's the issue
                    if strategy_name != 'YouTubeStrategy':
                        strategy_hold_reasons[f'wrong_strategy_{strategy_name}'] = strategy_hold_reasons.get(f'wrong_strategy_{strategy_name}', 0) + 1
                        if youtube_hold_count <= 3:
                            LOGGER.warning("  WARNING: Strategy is '%s', not 'YouTubeStrategy'!", strategy_name)
                    elif '11:00' in rationale or '11:00 AM' in rationale or 'before 11' in rationale_lower:
                        # Already filtered, shouldn't reach here, but track anyway
                        strategy_hold_reasons['before_11am'] = strategy_hold_reasons.get('before_11am', 0) + 1
                    elif 'oi' in rationale_lower and 'pcr' in rationale_lower and ('align' in rationale_lower or 'mismatch' in rationale_lower or 'do not align' in rationale_lower):
                        oi_pcr_mismatch_count += 1
                        strategy_hold_reasons['oi_pcr_mismatch'] = strategy_hold_reasons.get('oi_pcr_mismatch', 0) + 1
                    elif 'vwap' in rationale_lower or 'too far from vwap' in rationale_lower:
                        vwap_far_count += 1
                        strategy_hold_reasons['vwap_far'] = strategy_hold_reasons.get('vwap_far', 0) + 1
                    elif 'no clear oi' in rationale_lower or 'no clear direction' in rationale_lower or 'waiting for confirmation' in rationale_lower or 'no clear oi direction' in rationale_lower:
                        strategy_hold_reasons['no_oi_direction'] = strategy_hold_reasons.get('no_oi_direction', 0) + 1
                    elif 'pcr neutral' in rationale_lower or ('pcr' in rationale_lower and ('neutral' in rationale_lower or 'waiting' in rationale_lower)):
                        strategy_hold_reasons['pcr_neutral'] = strategy_hold_reasons.get('pcr_neutral', 0) + 1
                    else:
                        strategy_hold_reasons['other'] = strategy_hold_reasons.get('other', 0) + 1
                        # Log the actual rationale for "other" cases
                        if youtube_hold_count <= 5:
                            LOGGER.warning("  Other HOLD reason (full rationale): %s", rationale)
                    
                    signal_count['HOLD'] += 1
                    continue
                
                signal = youtube_recommendation.signal
                confidence = float(youtube_recommendation.confidence)
                rationale = youtube_recommendation.rationale
                
                signal_count[signal] = signal_count.get(signal, 0) + 1
                
                if signal == 'HOLD':
                    continue
                
                if confidence < self.config.min_confidence:
                    low_confidence_count += 1
                    continue
                
                direction = 1 if signal == 'BUY' else -1 if signal == 'SELL' else 0
                if direction == 0:
                    continue
                
                # Risk management - use fixed position sizing for YouTube strategy
                # YouTube strategy uses fixed risk per trade
                from risk_manager import get_optimal_position_size
                risk = get_optimal_position_size(
                    ml_confidence=confidence,
                    win_rate=0.6,  # YouTube strategy target win rate
                    avg_win_loss_ratio=1.5,  # 35/22.5 = 1.56
                    max_risk=self.config.max_risk_per_trade,
                    account_size=self.config.account_size,
                    margin_per_lot=self.config.margin_per_lot,
                )
                capital_allocated = risk.get('capital_allocated', 0.0)
                if capital_allocated <= 0.0:
                    zero_capital_count += 1
                    continue
                
                # Extract option type from metadata first (needed for P&L calculation)
                option_type = youtube_recommendation.metadata.get('option_type', 'UNKNOWN')
                target_points = youtube_recommendation.metadata.get('target_points', self.target_points)
                stop_loss_points = youtube_recommendation.metadata.get('stop_loss_points', self.stop_loss_points)
                
                # Calculate actual P&L based on target/stop-loss for YouTube strategy
                # For options: track if price hits target or stop-loss within holding period
                entry_price = features.get('underlying_price', 0.0)
                if entry_price <= 0:
                    # Skip if no valid entry price
                    continue
                
                # Initialize exit variables
                exit_price = entry_price
                exit_reason = 'time_limit'
                points_pnl = 0.0
                
                # Calculate target and stop-loss levels
                # For CALL: profit if price goes up, loss if price goes down
                # For PUT: profit if price goes down, loss if price goes up
                if option_type == 'CALL':
                    target_price = entry_price + target_points
                    stop_loss_price = entry_price - stop_loss_points
                elif option_type == 'PUT':
                    target_price = entry_price - target_points
                    stop_loss_price = entry_price + stop_loss_points
                else:
                    # Fallback: use direction
                    if direction > 0:  # BUY
                        target_price = entry_price + target_points
                        stop_loss_price = entry_price - stop_loss_points
                    else:  # SELL
                        target_price = entry_price - target_points
                        stop_loss_price = entry_price + stop_loss_points
                
                # Look ahead in the data to see which level is hit first
                # current_idx is already available from enumerate
                if current_idx >= len(frame_reset) - 1:
                    # Fallback to simple future_return if we can't look ahead
                    try:
                        future_return = float(row['future_return']) if 'future_return' in row.index else 0.0
                        if pd.isna(future_return):
                            future_return = 0.0
                    except (KeyError, ValueError, TypeError):
                        future_return = 0.0
                    # Use simple P&L calculation for fallback
                    # For options, approximate: 1% price move ≈ 1 point for ATM options
                    # Simplified: use future_return directly
                    gross_pnl = direction * future_return * capital_allocated
                    # Set exit variables for consistency
                    exit_price = entry_price * (1 + future_return) if future_return != 0 else entry_price
                    exit_reason = 'time_limit'
                    points_pnl = future_return * entry_price if entry_price > 0 else 0.0
                else:
                    # Look ahead up to holding_period_minutes
                    holding_period = self.config.holding_period_minutes
                    future_rows = frame_reset.iloc[current_idx+1:current_idx+1+holding_period] if current_idx+1 < len(frame_reset) else pd.DataFrame()
                    
                    # Reset exit variables for this branch (they were initialized above, but reset here for clarity)
                    exit_reason = 'time_limit'  # Default: didn't hit target or stop
                    exit_price = entry_price  # Default: no movement
                    points_pnl = 0.0
                    
                    # Check each future row to see if target or stop is hit
                    for future_idx, future_row in future_rows.iterrows():
                        try:
                            # Extract price from future row
                            if 'underlying_price' in future_row.index:
                                future_price_val = future_row['underlying_price']
                                # Handle Series/DataFrame
                                if isinstance(future_price_val, pd.Series):
                                    future_price_val = future_price_val.iloc[0] if len(future_price_val) > 0 else entry_price
                                elif isinstance(future_price_val, pd.DataFrame):
                                    future_price_val = future_price_val.iloc[0, 0] if future_price_val.shape[0] > 0 and future_price_val.shape[1] > 0 else entry_price
                                future_price = float(future_price_val) if not pd.isna(future_price_val) and future_price_val > 0 else entry_price
                            else:
                                future_price = entry_price
                            
                            if pd.isna(future_price) or future_price <= 0:
                                continue
                            
                            # Check if target is hit
                            if option_type == 'CALL' or (option_type == 'UNKNOWN' and direction > 0):
                                if future_price >= target_price:
                                    exit_price = target_price
                                    exit_reason = 'target_hit'
                                    points_pnl = target_points
                                    break
                                elif future_price <= stop_loss_price:
                                    exit_price = stop_loss_price
                                    exit_reason = 'stop_loss_hit'
                                    points_pnl = -stop_loss_points
                                    break
                            elif option_type == 'PUT':
                                if future_price <= target_price:
                                    exit_price = target_price
                                    exit_reason = 'target_hit'
                                    points_pnl = target_points
                                    break
                                elif future_price >= stop_loss_price:
                                    exit_price = stop_loss_price
                                    exit_reason = 'stop_loss_hit'
                                    points_pnl = -stop_loss_points
                                    break
                        except (KeyError, ValueError, TypeError, IndexError) as e:
                            LOGGER.debug(f"Error processing future row: {e}")
                            continue
                    
                    # If neither target nor stop was hit, calculate P&L based on final price
                    if exit_reason == 'time_limit':
                        if len(future_rows) > 0:
                            try:
                                last_row = future_rows.iloc[-1]
                                if 'underlying_price' in last_row.index:
                                    exit_price_val = last_row['underlying_price']
                                    # Handle Series/DataFrame
                                    if isinstance(exit_price_val, pd.Series):
                                        exit_price_val = exit_price_val.iloc[0] if len(exit_price_val) > 0 else entry_price
                                    elif isinstance(exit_price_val, pd.DataFrame):
                                        exit_price_val = exit_price_val.iloc[0, 0] if exit_price_val.shape[0] > 0 and exit_price_val.shape[1] > 0 else entry_price
                                    exit_price = float(exit_price_val) if not pd.isna(exit_price_val) and exit_price_val > 0 else entry_price
                                else:
                                    exit_price = entry_price
                            except (KeyError, ValueError, TypeError, IndexError):
                                exit_price = entry_price
                        else:
                            exit_price = entry_price
                        
                        # Calculate points P&L based on price movement
                        if option_type == 'CALL' or (option_type == 'UNKNOWN' and direction > 0):
                            points_pnl = exit_price - entry_price
                            # Cap at target or stop-loss
                            if points_pnl > target_points:
                                points_pnl = target_points
                                exit_reason = 'target_hit'
                            elif points_pnl < -stop_loss_points:
                                points_pnl = -stop_loss_points
                                exit_reason = 'stop_loss_hit'
                        elif option_type == 'PUT':
                            points_pnl = entry_price - exit_price
                            # Cap at target or stop-loss
                            if points_pnl > target_points:
                                points_pnl = target_points
                                exit_reason = 'target_hit'
                            elif points_pnl < -stop_loss_points:
                                points_pnl = -stop_loss_points
                                exit_reason = 'stop_loss_hit'
                    
                    # Calculate P&L in rupees
                    # For Nifty options: 1 point = 75 rupees per lot
                    # capital_allocated is already in rupees, so we need to convert points to rupees
                    # Assuming 1 lot = 75 points value, and capital_allocated represents position value
                    lot_size = 75  # Nifty lot size
                    points_to_rupees = lot_size
                    
                    # Calculate gross P&L: points_pnl * lot_size * number_of_lots
                    # number_of_lots = capital_allocated / (entry_price * lot_size) approximately
                    # Simplified: assume 1 point movement = proportional P&L based on capital
                    # For options, P&L is roughly: points_pnl * lot_size * (capital_allocated / margin_per_lot)
                    recommended_lots = risk.get('recommended_lots', 1)
                    gross_pnl = points_pnl * lot_size * recommended_lots
                    
                    # Calculate future_return for compatibility (as percentage)
                    future_return = points_pnl / entry_price if entry_price > 0 else 0.0
                
                # Both branches above should have set gross_pnl and future_return
                # At this point, both variables should be defined from the if/else above
                # No need to check locals() - variables are in the same scope
                
                transaction_cost = abs(capital_allocated) * total_cost_rate
                net_pnl = gross_pnl - transaction_cost
                
                predictions.append(direction)
                actual_returns.append(future_return)
                position_sizes.append(capital_allocated)
                net_pnls.append(net_pnl)
                
                gross_equity += gross_pnl
                net_equity += net_pnl
                
                trade = TradeRecord(
                    timestamp=timestamp_str,
                    signal=signal,
                    direction=direction,
                    confidence=confidence,
                    rationale=rationale,
                    future_return=future_return,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    transaction_cost=transaction_cost,
                    capital_allocated=capital_allocated,
                    position_fraction=risk.get('position_fraction', 0.0),
                    recommended_lots=risk.get('recommended_lots', 0),
                    metadata={
                        'strategy': 'youtube',
                        'option_type': option_type,
                        'target_points': target_points,
                        'stop_loss_points': stop_loss_points,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'points_pnl': points_pnl,
                        'oi_direction': youtube_recommendation.metadata.get('oi_direction'),
                        'pcr_direction': youtube_recommendation.metadata.get('pcr_direction'),
                        'pcr_value': youtube_recommendation.metadata.get('pcr_value'),
                        'vwap_aligned': youtube_recommendation.metadata.get('vwap_aligned'),
                        'atm_strike': youtube_recommendation.metadata.get('atm_strike'),
                        # Store volume features for analysis
                        'itm_volume_ce_change': features.get('itm_volume_ce_pct_change_3m_wavg', 0.0),
                        'itm_volume_pe_change': features.get('itm_volume_pe_pct_change_3m_wavg', 0.0),
                        'volume_call_total': features.get('nse_next_volume_call_total', features.get('total_volume_call', 0.0)),
                        'volume_put_total': features.get('nse_next_volume_put_total', features.get('total_volume_put', 0.0)),
                    }
                )
                trades.append(trade)
                
                equity_curve.append({
                    'timestamp': timestamp_str,
                    'gross_equity': gross_equity,
                    'net_equity': net_equity,
                    'num_trades': len(trades)
                })
        
        except Exception as e:
            LOGGER.error(f"Error during backtest: {e}", exc_info=True)
        
        # Calculate metrics
        from risk_manager import calculate_trading_metrics
        if trades:
            metrics = calculate_trading_metrics(
                predictions=predictions,
                actual_returns=actual_returns,
                position_sizes=position_sizes
            )
            metrics['num_trades'] = len(trades)
            metrics['gross_total_pnl'] = gross_equity
            metrics['net_total_pnl'] = net_equity
            metrics['total_transaction_cost'] = sum(t.transaction_cost for t in trades)
            
            # YouTube strategy specific metrics
            metrics['youtube_before_11am_filtered'] = before_11am_count
            metrics['youtube_hold_count'] = youtube_hold_count
            metrics['youtube_oi_pcr_mismatch'] = oi_pcr_mismatch_count
            metrics['youtube_vwap_far'] = vwap_far_count
            metrics['youtube_low_confidence'] = low_confidence_count
            metrics['youtube_zero_capital'] = zero_capital_count
            metrics['youtube_signal_breakdown'] = signal_count
            metrics['youtube_hold_reasons'] = strategy_hold_reasons
            
            # Calculate average target/stop from trades
            if trades:
                avg_target = np.mean([t.metadata.get('target_points', self.target_points) for t in trades])
                avg_stop = np.mean([t.metadata.get('stop_loss_points', self.stop_loss_points) for t in trades])
                metrics['youtube_avg_target_points'] = avg_target
                metrics['youtube_avg_stop_points'] = avg_stop
                
                # Enhanced reporting: Win/Loss breakdown
                winning_trades = [t for t in trades if t.gross_pnl > 0]
                losing_trades = [t for t in trades if t.gross_pnl < 0]
                breakeven_trades = [t for t in trades if t.gross_pnl == 0]
                
                metrics['youtube_winning_trades'] = len(winning_trades)
                metrics['youtube_losing_trades'] = len(losing_trades)
                metrics['youtube_breakeven_trades'] = len(breakeven_trades)
                metrics['youtube_win_rate'] = len(winning_trades) / len(trades) if trades else 0.0
                
                # Exit reason breakdown
                exit_reasons = {}
                for t in trades:
                    reason = t.metadata.get('exit_reason', 'unknown')
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                metrics['youtube_exit_reasons'] = exit_reasons
                
                # Points P&L statistics
                points_pnls = [t.metadata.get('points_pnl', 0.0) for t in trades]
                if points_pnls:
                    metrics['youtube_avg_points_pnl'] = np.mean(points_pnls)
                    metrics['youtube_total_points_pnl'] = sum(points_pnls)
                    metrics['youtube_max_points_win'] = max(points_pnls) if points_pnls else 0.0
                    metrics['youtube_max_points_loss'] = min(points_pnls) if points_pnls else 0.0
                
                # Option type breakdown
                call_trades = [t for t in trades if t.metadata.get('option_type') == 'CALL']
                put_trades = [t for t in trades if t.metadata.get('option_type') == 'PUT']
                metrics['youtube_call_trades'] = len(call_trades)
                metrics['youtube_put_trades'] = len(put_trades)
                metrics['youtube_call_pnl'] = sum(t.gross_pnl for t in call_trades)
                metrics['youtube_put_pnl'] = sum(t.gross_pnl for t in put_trades)
        else:
            metrics = {
                'num_trades': 0,
                'gross_total_pnl': 0.0,
                'net_total_pnl': 0.0,
                'youtube_before_11am_filtered': before_11am_count,
                'youtube_hold_count': youtube_hold_count,
                'youtube_oi_pcr_mismatch': oi_pcr_mismatch_count,
                'youtube_vwap_far': vwap_far_count,
                'youtube_low_confidence': low_confidence_count,
                'youtube_zero_capital': zero_capital_count,
                'youtube_signal_breakdown': signal_count,
                'youtube_hold_reasons': strategy_hold_reasons
            }
        
        result = BacktestResult(self.config, trades, metrics, equity_curve, raw_rows=len(frame))
        LOGGER.warning("YouTube Strategy backtest completed: %d trades generated", len(trades))
        LOGGER.info("Filtered: %d before 11 AM, %d YouTube HOLD, %d OI/PCR mismatch, %d VWAP far",
                   before_11am_count, youtube_hold_count, oi_pcr_mismatch_count, vwap_far_count)
        LOGGER.info("Additional filters: %d low confidence, %d zero capital", 
                   low_confidence_count, zero_capital_count)
        LOGGER.info("Signal breakdown: %s", signal_count)
        if strategy_hold_reasons:
            LOGGER.info("HOLD reasons: %s", strategy_hold_reasons)
        
        return result
    
    def _prepare_frame(self) -> pd.DataFrame:
        """Load and prepare historical data."""
        try:
            raw = db.load_historical_data_for_ml(self.config.exchange, self.config.start, self.config.end)
        except Exception as exc:
            LOGGER.error("Failed to load historical data: %s", exc, exc_info=True)
            return pd.DataFrame()
        
        if raw is None or raw.empty:
            LOGGER.warning("No raw data loaded for %s between %s and %s", 
                          self.config.exchange, self.config.start, self.config.end)
            return pd.DataFrame()
        
        # Filter to only include data after 11:00 AM (optional, for efficiency)
        # But we'll do the check in the loop to be precise
        
        return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest YouTube Strategy over historical data.")
    parser.add_argument("--exchange", required=True, choices=["NSE", "BSE"])
    parser.add_argument("--start", required=True, type=_parse_date, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, type=_parse_date, help="End date (YYYY-MM-DD).")
    parser.add_argument("--holding-period", type=int, default=15, help="Holding window in minutes.")
    parser.add_argument("--cost-bps", type=float, default=2.0, help="Transaction cost in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in basis points.")
    parser.add_argument("--min-confidence", type=float, default=0.65, 
                       help="Minimum confidence to trade (YouTube strategy is conservative, default 0.65 for better win rate).")
    parser.add_argument("--max-trades", type=int, default=None, help="Upper bound on number of trades.")
    parser.add_argument("--account-size", type=float, default=1_000_000.0, help="Account notional in INR.")
    parser.add_argument("--margin-per-lot", type=float, default=75_000.0, help="Margin per index lot.")
    parser.add_argument("--max-risk", type=float, default=0.02, help="Max risk per trade (fraction).")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional cap on rows for dry-runs.")
    parser.add_argument("--output", type=Path, default=None, help="Path to dump JSON results.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), 
                       format="%(asctime)s - %(levelname)s - %(message)s")
    
    config = BacktestConfig(
        exchange=args.exchange,
        start=args.start,
        end=args.end,
        strategy="youtube",
        holding_period_minutes=args.holding_period,
        transaction_cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_confidence=args.min_confidence,
        max_trades=args.max_trades,
        account_size=args.account_size,
        margin_per_lot=args.margin_per_lot,
        max_risk_per_trade=args.max_risk,
        limit_rows=args.limit_rows,
        use_rl=True,  # Enable router
    )
    
    engine = YouTubeStrategyBacktestEngine(config)
    result = engine.run()
    
    if not result.metrics:
        logging.warning("No metrics returned. Ensure models exist and the date range has data.")
    else:
        metrics = result.metrics
        logging.info("YouTube Strategy Backtest Results:")
        logging.info("  Trades: %d", metrics.get("num_trades", 0))
        logging.info("  Net PnL: ₹%.2f", metrics.get("net_total_pnl", 0.0))
        logging.info("  Win Rate: %.2f%%", metrics.get("win_rate", 0.0) * 100)
        logging.info("  Sharpe Ratio: %.2f", metrics.get("sharpe_ratio", 0.0))
        logging.info("  Filtered before 11 AM: %d", metrics.get("youtube_before_11am_filtered", 0))
        logging.info("  YouTube HOLD signals: %d", metrics.get("youtube_hold_count", 0))
        logging.info("  OI/PCR mismatches: %d", metrics.get("youtube_oi_pcr_mismatch", 0))
        logging.info("  VWAP too far: %d", metrics.get("youtube_vwap_far", 0))
        logging.info("  Low confidence: %d", metrics.get("youtube_low_confidence", 0))
        logging.info("  Zero capital: %d", metrics.get("youtube_zero_capital", 0))
        if metrics.get("youtube_hold_reasons"):
            logging.info("  HOLD reasons breakdown:")
            for reason, count in metrics.get("youtube_hold_reasons", {}).items():
                logging.info("    - %s: %d", reason, count)
        logging.info("")
        logging.info("NOTE: If all signals are HOLD, check:")
        logging.info("  1. Are OI change features populated? (nse_next_oi_change_call_total, nse_next_oi_change_put_total)")
        logging.info("  2. Are PCR values available? (pcr_total_volume or pcr_total_oi)")
        logging.info("  3. See YOUTUBE_STRATEGY_BACKTEST_DIAGNOSTICS.md for diagnostic queries")
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_dict = _sanitize_for_json(result.to_dict())
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result_dict, handle, indent=2)
        logging.info("Saved backtest report to %s", args.output)
    else:
        # Auto-save to reports directory
        backtest_dir = Path('reports') / 'backtests'
        backtest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = backtest_dir / f'youtube_strategy_{args.exchange}_{timestamp}.json'
        try:
            result_dict = _sanitize_for_json(result.to_dict())
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(result_dict, handle, indent=2)
            logging.warning("Saved backtest report to: %s", output_file)
        except Exception as e:
            logging.error("Failed to save backtest report: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
