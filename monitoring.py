"""
Blueprint providing model monitoring views and APIs.
Enhanced with Phase 2 metrics monitoring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from flask import Blueprint, jsonify, render_template, request

from strings import UI_STRINGS
from metrics.phase2_metrics import get_metrics_collector

monitoring_bp = Blueprint('monitoring', __name__)
EXCHANGES = ['NSE', 'BSE']
BACKTEST_DIR = Path('reports') / 'backtests'
ONLINE_STATE_FILE = Path('reports') / 'online_learning_state.json'


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
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

