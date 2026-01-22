"""
Blueprint providing model monitoring views and APIs.
Enhanced with Phase 2 metrics monitoring.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Any

from flask import Blueprint, jsonify, render_template, request

from strings import UI_STRINGS
from metrics.phase2_metrics import get_metrics_collector

monitoring_bp = Blueprint('monitoring', __name__)
EXCHANGES = ['NSE', 'BSE']
BACKTEST_DIR = Path('reports') / 'backtests'
ONLINE_STATE_FILE = Path('reports') / 'online_learning_state.json'


def _sanitize_nan(obj: Any) -> Any:
    """
    Recursively replace NaN, Infinity values with None (which becomes null in JSON).
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_nan(item) for item in obj]
    else:
        return obj


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding='utf-8')
        # Replace NaN, Infinity, -Infinity with null before parsing
        # This handles cases where JSON was written with NaN (invalid JSON)
        content = re.sub(r'\bNaN\b', 'null', content)
        content = re.sub(r'\bInfinity\b', 'null', content)
        content = re.sub(r'\b-Infinity\b', 'null', content)
        data = json.loads(content)
        # Sanitize any remaining NaN values that might have been parsed as strings
        return _sanitize_nan(data)
    except json.JSONDecodeError as e:
        # Log the error for debugging but return empty dict
        import logging
        logging.warning(f"Failed to parse JSON from {path}: {e}")
        return {}


def _load_health_payload() -> Dict[str, Dict]:
    online_state = _load_json(ONLINE_STATE_FILE)
    payload: Dict[str, Dict] = {}
    for exchange in EXCHANGES:
        reports_dir = Path('models') / exchange / 'reports'
        training_report = _load_json(reports_dir / 'training_report.json')
        auto_ml_summary = _load_json(reports_dir / 'auto_ml_summary.json')
        backtest_report = _load_backtest_summary(exchange)
        online_stats = online_state.get(exchange, {})

        payload[exchange] = {
            'training': training_report,
            'auto_ml': auto_ml_summary,
            'backtest': backtest_report,
            'online_learning': online_stats,
        }
    return payload


def _load_backtest_summary(exchange: str) -> Dict:
    if BACKTEST_DIR.exists():
        primary = BACKTEST_DIR / f'{exchange.upper()}.json'
        if primary.exists():
            return _load_json(primary)
    # Fallback: check models/<exchange>/reports/backtest_summary.json
    fallback = Path('models') / exchange / 'reports' / 'backtest_summary.json'
    return _load_json(fallback)


@monitoring_bp.route('/monitoring')
def monitoring_dashboard():
    return render_template('monitoring.html', strings=UI_STRINGS)


@monitoring_bp.route('/monitoring/api/model-health')
def monitoring_api():
    return jsonify(_load_health_payload())


@monitoring_bp.route('/monitoring/api/phase2-metrics')
def phase2_metrics_api():
    """API endpoint for Phase 2 validation metrics."""
    exchange = request.args.get('exchange', 'NSE')
    hours = int(request.args.get('hours', 24))
    
    if exchange not in EXCHANGES:
        return jsonify({'error': 'Invalid exchange'}), 400
    
    collector = get_metrics_collector(exchange)
    summary = collector.compute_summary(hours=hours)
    
    return jsonify(summary)


@monitoring_bp.route('/monitoring/api/phase2-metrics/<metric_type>')
def phase2_metrics_detail_api(metric_type: str):
    """API endpoint for detailed Phase 2 metrics."""
    exchange = request.args.get('exchange', 'NSE')
    limit = int(request.args.get('limit', 100))
    
    if exchange not in EXCHANGES:
        return jsonify({'error': 'Invalid exchange'}), 400
    
    valid_types = ['data_quality', 'model_performance', 'system_health', 'paper_trading']
    if metric_type not in valid_types:
        return jsonify({'error': f'Invalid metric type. Must be one of: {valid_types}'}), 400
    
    collector = get_metrics_collector(exchange)
    metrics = collector.get_recent_metrics(metric_type, limit=limit)
    
    return jsonify({
        'exchange': exchange,
        'metric_type': metric_type,
        'count': len(metrics),
        'metrics': metrics
    })

