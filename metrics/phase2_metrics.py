"""
Phase 2 Metrics Collector for validation period monitoring.

Tracks data quality, model performance, system health, and paper trading metrics.
"""
from __future__ import annotations

import json
import logging
import os
import psutil
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import database_new as db
from time_utils import now_ist

LOGGER = logging.getLogger(__name__)

METRICS_DIR = Path("metrics") / "phase2_metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class Phase2MetricsCollector:
    """Collects and aggregates Phase 2 validation metrics."""
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.metrics_file = METRICS_DIR / f"{exchange}_metrics.jsonl"
        self.summary_file = METRICS_DIR / f"{exchange}_summary.json"
        
        # In-memory buffers for real-time metrics
        self.data_quality_buffer: deque = deque(maxlen=1000)
        self.model_performance_buffer: deque = deque(maxlen=500)
        self.system_health_buffer: deque = deque(maxlen=200)
        self.paper_trading_buffer: deque = deque(maxlen=1000)
        
        # Initialize summary
        self._initialize_summary()
    
    def _initialize_summary(self) -> None:
        """Initialize or load existing summary."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    self.summary = json.load(f)
            except Exception:
                self.summary = self._create_empty_summary()
        else:
            self.summary = self._create_empty_summary()
    
    def _create_empty_summary(self) -> Dict[str, Any]:
        """Create empty summary structure."""
        return {
            'exchange': self.exchange,
            'last_updated': now_ist().isoformat(),
            'data_quality': {
                'macro_availability_pct': 0.0,
                'depth_capture_success_rate': 0.0,
                'feature_engineering_errors': 0,
                'database_write_success_rate': 0.0,
            },
            'model_performance': {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0.0,
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'signal_frequency_per_hour': 0.0,
            },
            'system_health': {
                'memory_usage_mb': 0.0,
                'cpu_usage_pct': 0.0,
                'database_size_mb': 0.0,
                'error_rate': 0.0,
                'uptime_hours': 0.0,
            },
            'paper_trading': {
                'total_executions': 0,
                'successful_executions': 0,
                'rejected_executions': 0,
                'rejection_reasons': {},
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'portfolio_constraint_violations': 0,
            },
        }
    
    def record_data_quality(
        self,
        macro_available: bool,
        depth_captured: bool,
        feature_engineering_success: bool,
        db_write_success: bool,
    ) -> None:
        """Record data quality metrics."""
        entry = {
            'timestamp': now_ist().isoformat(),
            'macro_available': macro_available,
            'depth_captured': depth_captured,
            'feature_engineering_success': feature_engineering_success,
            'db_write_success': db_write_success,
        }
        self.data_quality_buffer.append(entry)
        self._append_to_file('data_quality', entry)
    
    def record_model_performance(
        self,
        signal: str,
        confidence: float,
        source: str = 'lightgbm',
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record model performance metrics."""
        entry = {
            'timestamp': now_ist().isoformat(),
            'signal': signal,
            'confidence': confidence,
            'source': source,
            'metadata': metadata or {},
        }
        self.model_performance_buffer.append(entry)
        self._append_to_file('model_performance', entry)
    
    def record_system_health(
        self,
        memory_mb: Optional[float] = None,
        cpu_pct: Optional[float] = None,
        db_size_mb: Optional[float] = None,
        error_count: int = 0,
    ) -> None:
        """Record system health metrics."""
        if memory_mb is None:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        
        if cpu_pct is None:
            cpu_pct = psutil.cpu_percent(interval=0.1)
        
        if db_size_mb is None:
            db_size_mb = self._get_database_size()
        
        entry = {
            'timestamp': now_ist().isoformat(),
            'memory_mb': memory_mb,
            'cpu_pct': cpu_pct,
            'db_size_mb': db_size_mb,
            'error_count': error_count,
        }
        self.system_health_buffer.append(entry)
        self._append_to_file('system_health', entry)
    
    def record_paper_trading(
        self,
        executed: bool,
        reason: str,
        signal: str,
        confidence: float,
        quantity_lots: int = 0,
        pnl: Optional[float] = None,
        constraint_violation: bool = False,
    ) -> None:
        """Record paper trading execution metrics."""
        ts = now_ist()
        entry = {
            'timestamp': ts.isoformat(),
            'executed': executed,
            'reason': reason,
            'signal': signal,
            'confidence': confidence,
            'quantity_lots': quantity_lots,
            'pnl': pnl,
            'constraint_violation': constraint_violation,
        }
        self.paper_trading_buffer.append(entry)
        self._append_to_file('paper_trading', entry)
        # Also persist to the relational database for long-term analytics
        try:
            db.record_paper_trading_metric(
                exchange=self.exchange,
                timestamp=ts,
                executed=executed,
                reason=reason,
                signal=signal,
                confidence=confidence,
                quantity_lots=quantity_lots,
                pnl=pnl,
                constraint_violation=constraint_violation,
            )
        except Exception as exc:
            LOGGER.debug(f"Failed to persist paper trading metric to DB: {exc}")
    
    def _append_to_file(self, metric_type: str, entry: Dict) -> None:
        """Append metric entry to JSONL file."""
        try:
            log_entry = {
                'type': metric_type,
                'exchange': self.exchange,
                **entry
            }
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            LOGGER.debug(f"Failed to write metric: {e}")
    
    def _get_database_size(self) -> float:
        """Get database file size in MB."""
        try:
            db_path = Path("db_new.db")
            if db_path.exists():
                return db_path.stat().st_size / 1024 / 1024
        except Exception:
            pass
        return 0.0
    
    def compute_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Compute summary statistics for the last N hours."""
        cutoff = datetime.fromisoformat(now_ist().isoformat()) - timedelta(hours=hours)
        
        # Data Quality
        dq_recent = [
            e for e in self.data_quality_buffer
            if datetime.fromisoformat(e['timestamp']) >= cutoff
        ]
        if dq_recent:
            dq_summary = {
                'macro_availability_pct': sum(e['macro_available'] for e in dq_recent) / len(dq_recent) * 100,
                'depth_capture_success_rate': sum(e['depth_captured'] for e in dq_recent) / len(dq_recent) * 100,
                'feature_engineering_errors': sum(not e['feature_engineering_success'] for e in dq_recent),
                'database_write_success_rate': sum(e['db_write_success'] for e in dq_recent) / len(dq_recent) * 100,
            }
        else:
            dq_summary = self.summary['data_quality']
        
        # Model Performance
        mp_recent = [
            e for e in self.model_performance_buffer
            if datetime.fromisoformat(e['timestamp']) >= cutoff
        ]
        if mp_recent:
            confidences = [e['confidence'] for e in mp_recent]
            signals_by_type = defaultdict(int)
            for e in mp_recent:
                signals_by_type[e['signal']] += 1
            
            mp_summary = {
                'total_signals': len(mp_recent),
                'buy_signals': signals_by_type.get('BUY', 0),
                'sell_signals': signals_by_type.get('SELL', 0),
                'hold_signals': signals_by_type.get('HOLD', 0),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'confidence_distribution': {
                    'high': sum(1 for c in confidences if c >= 0.7),
                    'medium': sum(1 for c in confidences if 0.5 <= c < 0.7),
                    'low': sum(1 for c in confidences if c < 0.5),
                },
                'signal_frequency_per_hour': len(mp_recent) / max(hours, 1),
            }
        else:
            mp_summary = self.summary['model_performance']
        
        # System Health
        sh_recent = [
            e for e in self.system_health_buffer
            if datetime.fromisoformat(e['timestamp']) >= cutoff
        ]
        if sh_recent:
            sh_summary = {
                'memory_usage_mb': float(np.mean([e['memory_mb'] for e in sh_recent])),
                'cpu_usage_pct': float(np.mean([e['cpu_pct'] for e in sh_recent])),
                'database_size_mb': sh_recent[-1]['db_size_mb'] if sh_recent else 0.0,
                'error_rate': sum(e['error_count'] for e in sh_recent) / max(hours, 1),
                'uptime_hours': hours,  # Simplified
            }
        else:
            sh_summary = self.summary['system_health']
        
        # Paper Trading
        pt_recent = [
            e for e in self.paper_trading_buffer
            if datetime.fromisoformat(e['timestamp']) >= cutoff
        ]
        if pt_recent:
            executed_trades = [e for e in pt_recent if e['executed']]
            rejected_trades = [e for e in pt_recent if not e['executed']]
            
            rejection_reasons = defaultdict(int)
            for e in rejected_trades:
                rejection_reasons[e['reason']] += 1
            
            pnls = [e['pnl'] for e in executed_trades if e.get('pnl') is not None]
            
            pt_summary = {
                'total_executions': len(executed_trades),
                'successful_executions': len(executed_trades),
                'rejected_executions': len(rejected_trades),
                'rejection_reasons': dict(rejection_reasons),
                'total_pnl': float(sum(pnls)) if pnls else 0.0,
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0.0,
                'avg_trade_pnl': float(np.mean(pnls)) if pnls else 0.0,
                'portfolio_constraint_violations': sum(e['constraint_violation'] for e in pt_recent),
            }
        else:
            pt_summary = self.summary['paper_trading']
        
        summary = {
            'exchange': self.exchange,
            'last_updated': now_ist().isoformat(),
            'period_hours': hours,
            'data_quality': dq_summary,
            'model_performance': mp_summary,
            'system_health': sh_summary,
            'paper_trading': pt_summary,
        }
        
        # Save summary
        self.summary = summary
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            LOGGER.debug(f"Failed to save summary: {e}")
        
        return summary
    
    def get_recent_metrics(self, metric_type: str, limit: int = 100) -> List[Dict]:
        """Get recent metrics of a specific type."""
        buffers = {
            'data_quality': self.data_quality_buffer,
            'model_performance': self.model_performance_buffer,
            'system_health': self.system_health_buffer,
            'paper_trading': self.paper_trading_buffer,
        }
        
        buffer = buffers.get(metric_type)
        if not buffer:
            return []
        
        return list(buffer)[-limit:]


# Global collectors
_metrics_collectors: Dict[str, Phase2MetricsCollector] = {}


def get_metrics_collector(exchange: str) -> Phase2MetricsCollector:
    """Get or create metrics collector for exchange."""
    if exchange not in _metrics_collectors:
        _metrics_collectors[exchange] = Phase2MetricsCollector(exchange)
    return _metrics_collectors[exchange]

