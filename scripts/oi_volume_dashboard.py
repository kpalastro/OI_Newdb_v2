#!/usr/bin/env python3
"""
Interactive ITM OI + Volume dashboard using TradingView Lightweight Charts.

Features:
- Select exchange (NSE / BSE)
- Select date range (from / to)
- Plots:
  - ITM CE / PE OI % change vs time
  - ITM CE / PE Volume % change vs time

Run:
    python scripts/oi_volume_dashboard.py

Then open in browser:
    http://127.0.0.1:5055/
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request, Response

import sys

# Ensure project root is on sys.path so database_new imports correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import database_new as db


app = Flask(__name__)


def _parse_date(s: str, default: date) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return default


def _parse_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


@app.route("/api/itm_oi_volume")
def api_itm_oi_volume() -> Response:
    """Return ITM CE/PE OI% and Volume% time series for given exchange and date range."""
    exchange = request.args.get("exchange", "NSE").upper()
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=5)
    start_date = _parse_date(request.args.get("start", ""), default_start)
    end_date = _parse_date(request.args.get("end", ""), today)

    # Convert to datetimes (UTC window)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    conn = db.get_db_connection()
    cur = conn.cursor()

    # Some deployments store volume deltas inside feature_payload JSONB
    query = """
        SELECT
            timestamp,
            itm_oi_ce_pct_change_3m_wavg AS ce_oi_pct,
            itm_oi_pe_pct_change_3m_wavg AS pe_oi_pct,
            feature_payload
        FROM ml_features
        WHERE exchange = %s
          AND timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp
    """
    cur.execute(query, (exchange, start_dt, end_dt))
    rows = cur.fetchall()
    db.release_db_connection(conn)

    points: List[Dict[str, Any]] = []
    for ts, ce_oi, pe_oi, payload in rows:
        ce_vol = None
        pe_vol = None
        if payload:
            try:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict):
                    ce_vol = payload.get("itm_volume_ce_pct_change_3m_wavg")
                    pe_vol = payload.get("itm_volume_pe_pct_change_3m_wavg")
            except Exception:
                pass

        # Use Unix timestamp in seconds for Lightweight Charts
        t = int(ts.timestamp())
        points.append(
            {
                "time": t,
                "ce_oi_pct": float(ce_oi) if ce_oi is not None else None,
                "pe_oi_pct": float(pe_oi) if pe_oi is not None else None,
                "ce_vol_pct": float(ce_vol) if ce_vol is not None else None,
                "pe_vol_pct": float(pe_vol) if pe_vol is not None else None,
            }
        )

    return jsonify(
        {
            "exchange": exchange,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "points": points,
        }
    )


@app.route("/api/bars_1m")
def api_bars_1m() -> Response:
    """
    Return 1-minute OHLCV (and OI) bars from multi_resolution_bars for a symbol.

    Params:
      - exchange: NSE/BSE
      - symbol: optional (if omitted, picks the most common symbol in range)
      - start/end: YYYY-MM-DD
      - limit: max number of rows (default 5000)
    """
    exchange = request.args.get("exchange", "NSE").upper()
    symbol = (request.args.get("symbol") or "").strip()
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=5)
    start_date = _parse_date(request.args.get("start", ""), default_start)
    end_date = _parse_date(request.args.get("end", ""), today)
    limit = _parse_int(request.args.get("limit", ""), 5000)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    conn = db.get_db_connection()
    cur = conn.cursor()

    # Resolution naming varies across deployments; accept common variants.
    res_variants = ("1m", "1", "1min", "minute", "MINUTE", "ONE_MINUTE")

    # If symbol not provided, pick a reasonable default: most frequent symbol in window.
    chosen_symbol = symbol
    if not chosen_symbol:
        cur.execute(
            """
            SELECT symbol, COUNT(*) AS c
            FROM multi_resolution_bars
            WHERE exchange = %s
              AND timestamp >= %s
              AND timestamp < %s
              AND resolution = ANY(%s)
              AND symbol IS NOT NULL
            GROUP BY symbol
            ORDER BY c DESC
            LIMIT 1
            """,
            (exchange, start_dt, end_dt, list(res_variants)),
        )
        row = cur.fetchone()
        if row and row[0]:
            chosen_symbol = row[0]

    bars: List[Dict[str, Any]] = []
    if chosen_symbol:
        cur.execute(
            """
            SELECT
              timestamp,
              open_price, high_price, low_price, close_price,
              volume, oi
            FROM multi_resolution_bars
            WHERE exchange = %s
              AND symbol = %s
              AND timestamp >= %s
              AND timestamp < %s
              AND resolution = ANY(%s)
            ORDER BY timestamp
            LIMIT %s
            """,
            (exchange, chosen_symbol, start_dt, end_dt, list(res_variants), limit),
        )
        rows = cur.fetchall()
        for ts, o, h, l, c, v, oi in rows:
            bars.append(
                {
                    "time": int(ts.timestamp()),
                    "open": float(o) if o is not None else None,
                    "high": float(h) if h is not None else None,
                    "low": float(l) if l is not None else None,
                    "close": float(c) if c is not None else None,
                    "volume": int(v) if v is not None else None,
                    "oi": int(oi) if oi is not None else None,
                }
            )

    db.release_db_connection(conn)
    return jsonify(
        {
            "exchange": exchange,
            "symbol": chosen_symbol,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "bars": bars,
        }
    )


@app.route("/api/trade_logs")
def api_trade_logs() -> Response:
    """
    Return trade log entries from CSVs for the selected date range.
    Used to overlay markers on the 1m chart.
    """
    exchange = request.args.get("exchange", "").upper().strip()
    symbol = (request.args.get("symbol") or "").strip()
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=5)
    start_date = _parse_date(request.args.get("start", ""), default_start)
    end_date = _parse_date(request.args.get("end", ""), today)

    # Load CSVs from trade_logs/trades_YYYY-MM-DD.csv
    rows: List[Dict[str, Any]] = []
    cur = start_date
    while cur <= end_date:
        p = PROJECT_ROOT / "trade_logs" / f"trades_{cur.strftime('%Y-%m-%d')}.csv"
        if p.exists():
            try:
                import pandas as pd

                df = pd.read_csv(p)
                # Normalize
                if "entry_timestamp" in df.columns:
                    df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"], errors="coerce")
                if "exit_timestamp" in df.columns:
                    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"], errors="coerce")

                for _, r in df.iterrows():
                    ex = str(r.get("exchange", "")).upper()
                    sym = str(r.get("symbol", "")).strip()
                    if exchange and ex != exchange:
                        continue
                    if symbol and sym != symbol:
                        continue

                    entry_ts = r.get("entry_timestamp")
                    exit_ts = r.get("exit_timestamp")
                    if pd.isna(entry_ts):
                        continue

                    side = str(r.get("side", "BUY")).upper()
                    pnl = r.get("pnl")
                    try:
                        pnl = float(pnl) if pnl is not None and pnl == pnl else None
                    except Exception:
                        pnl = None

                    rows.append(
                        {
                            "position_id": r.get("position_id"),
                            "symbol": sym,
                            "exchange": ex,
                            "side": side,
                            "entry_time": int(pd.Timestamp(entry_ts).timestamp()),
                            "exit_time": int(pd.Timestamp(exit_ts).timestamp()) if exit_ts == exit_ts else None,
                            "pnl": pnl,
                            "exit_reason": r.get("exit_reason"),
                        }
                    )
            except Exception:
                pass
        cur = cur + timedelta(days=1)

    return jsonify(
        {
            "exchange": exchange or None,
            "symbol": symbol or None,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "trades": rows,
        }
    )


@app.route("/api/paper_trading_signals")
def api_paper_trading_signals() -> Response:
    """
    Return paper trading signals from paper_trading_metrics table.
    Used to overlay BUY/SELL markers on the candlestick chart.
    """
    exchange = request.args.get("exchange", "").upper().strip()
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=5)
    start_date = _parse_date(request.args.get("start", ""), default_start)
    end_date = _parse_date(request.args.get("end", ""), today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    conn = db.get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT
            timestamp,
            signal,
            executed,
            confidence,
            reason,
            metadata
        FROM paper_trading_metrics
        WHERE timestamp >= %s
          AND timestamp < %s
    """
    params = [start_dt, end_dt]

    if exchange:
        query += " AND exchange = %s"
        params.append(exchange)

    query += " ORDER BY timestamp"

    cur.execute(query, params)
    rows = cur.fetchall()
    db.release_db_connection(conn)

    signals: List[Dict[str, Any]] = []
    for ts, signal, executed, confidence, reason, metadata in rows:
        signal_str = str(signal or "").upper().strip()
        if signal_str not in ("BUY", "SELL"):
            continue

        # Extract symbol from metadata if available
        symbol_from_meta = None
        if metadata:
            try:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if isinstance(metadata, dict):
                    symbol_from_meta = metadata.get("symbol")
            except Exception:
                pass

        signals.append(
            {
                "time": int(ts.timestamp()),
                "signal": signal_str,
                "executed": bool(executed) if executed is not None else False,
                "confidence": float(confidence) if confidence is not None else None,
                "reason": str(reason) if reason else None,
                "symbol": symbol_from_meta,
            }
        )

    return jsonify(
        {
            "exchange": exchange or None,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "signals": signals,
        }
    )


@app.route("/api/symbols")
def api_symbols() -> Response:
    """
    Return list of available symbols from multi_resolution_bars for the given exchange and date range.
    Used to populate the symbol dropdown.
    """
    exchange = request.args.get("exchange", "NSE").upper()
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=5)
    start_date = _parse_date(request.args.get("start", ""), default_start)
    end_date = _parse_date(request.args.get("end", ""), today)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    res_variants = ("1m", "1", "1min", "minute", "MINUTE", "ONE_MINUTE")

    conn = db.get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT symbol, COUNT(*) AS bar_count
        FROM multi_resolution_bars
        WHERE exchange = %s
          AND timestamp >= %s
          AND timestamp < %s
          AND resolution = ANY(%s)
          AND symbol IS NOT NULL
        GROUP BY symbol
        ORDER BY bar_count DESC, symbol ASC
        LIMIT 100
        """,
        (exchange, start_dt, end_dt, list(res_variants)),
    )
    rows = cur.fetchall()
    db.release_db_connection(conn)

    symbols = [{"symbol": row[0], "bar_count": row[1]} for row in rows]

    return jsonify(
        {
            "exchange": exchange,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "symbols": symbols,
        }
    )


@app.route("/")
def index() -> str:
    """Serve a single-page dashboard using TradingView Lightweight Charts."""
    # Inline HTML/JS for simplicity
    html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>ITM OI + Volume Dashboard</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 0; background: #0b1622; color: #e0e6f0; }
      .container { padding: 16px; max-width: 1200px; margin: 0 auto; }
      .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; align-items: center; }
      .controls label { font-size: 13px; margin-right: 4px; }
      .controls input, .controls select, .controls button { font-size: 13px; padding: 4px 6px; border-radius: 4px; border: 1px solid #2b3a4a; background: #111827; color: #e0e6f0; }
      .controls button { cursor: pointer; background: #2563eb; border-color: #2563eb; }
      .controls button:disabled { opacity: 0.5; cursor: default; }
      .chart-row { display: flex; flex-direction: column; gap: 8px; }
      .chart-title { font-size: 14px; margin-top: 8px; margin-bottom: 4px; }
      #chart-1m { height: 320px; }
      #chart-oi, #chart-vol { height: 260px; }
      .status { font-size: 12px; margin-top: 6px; color: #9ca3af; }
      a { color: #60a5fa; }
    </style>
    <!-- Pin a specific Lightweight Charts version for stable API -->
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
  </head>
  <body>
    <div class="container">
      <h2>ITM CE/PE OI &amp; Volume % Change (TradingView Lightweight Charts)</h2>
      <div class="controls">
        <label for="exchange">Exchange:</label>
        <select id="exchange">
          <option value="NSE">NSE</option>
          <option value="BSE">BSE</option>
        </select>
        <label for="symbol">Symbol:</label>
        <select id="symbol">
          <option value="">(Auto-select)</option>
        </select>
        <label for="start">From:</label>
        <input type="date" id="start" />
        <label for="end">To:</label>
        <input type="date" id="end" />
        <button id="load-btn">Load</button>
        <span class="status" id="status"></span>
      </div>
      <div class="chart-row">
        <div class="chart-title">1m Candles (multi_resolution_bars) + trade markers</div>
        <div id="chart-1m"></div>
        <div class="chart-title">ITM CE/PE OI % Change (3m wavg)</div>
        <div id="chart-oi"></div>
        <div class="chart-title">ITM CE/PE Volume % Change (3m wavg)</div>
        <div id="chart-vol"></div>
      </div>
    </div>
    <script>
      const statusEl = document.getElementById('status');
      const startInput = document.getElementById('start');
      const endInput = document.getElementById('end');
      const exchangeSelect = document.getElementById('exchange');
      const symbolSelect = document.getElementById('symbol');
      const loadBtn = document.getElementById('load-btn');

      // Load symbols dropdown when exchange/date range changes
      async function loadSymbols() {
        const ex = exchangeSelect.value || 'NSE';
        const start = startInput.value;
        const end = endInput.value;
        if (!start || !end) return;

        try {
          const params = new URLSearchParams({ exchange: ex, start, end });
          const resp = await fetch('/api/symbols?' + params.toString());
          if (!resp.ok) return;
          const data = await resp.json();
          const symbols = data.symbols || [];

          // Clear and repopulate dropdown
          symbolSelect.innerHTML = '<option value="">(Auto-select)</option>';
          for (const s of symbols) {
            const opt = document.createElement('option');
            opt.value = s.symbol;
            opt.textContent = s.symbol + (s.bar_count ? ` (${s.bar_count} bars)` : '');
            symbolSelect.appendChild(opt);
          }
        } catch (err) {
          console.debug('loadSymbols error:', err);
        }
      }

      // Reload symbols when exchange or dates change
      exchangeSelect.addEventListener('change', loadSymbols);
      startInput.addEventListener('change', loadSymbols);
      endInput.addEventListener('change', loadSymbols);

      // Default date range: last 5 days
      (function initDates() {
        const today = new Date();
        const endStr = today.toISOString().slice(0, 10);
        const start = new Date(today.getTime() - 4 * 24 * 60 * 60 * 1000);
        const startStr = start.toISOString().slice(0, 10);
        startInput.value = startStr;
        endInput.value = endStr;
        // Load symbols after dates are set
        setTimeout(loadSymbols, 100);
      })();

      // 1m candle chart
      const chart1m = LightweightCharts.createChart(document.getElementById('chart-1m'), {
        layout: { background: { color: '#0b1622' }, textColor: '#d1d5db' },
        grid: { vertLines: { color: '#1f2933' }, horzLines: { color: '#1f2933' } },
        timeScale: { borderColor: '#374151', timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderColor: '#374151' },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      });
      const candleSeries = chart1m.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#22c55e',
        borderDownColor: '#ef4444',
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      });

      // Create charts
      const chartOiContainer = document.getElementById('chart-oi');
      const chartOi = LightweightCharts.createChart(chartOiContainer, {
        layout: { background: { color: '#0b1622' }, textColor: '#d1d5db' },
        grid: {
          vertLines: { color: '#1f2933' },
          horzLines: { color: '#1f2933' },
        },
        timeScale: {
          borderColor: '#374151',
          timeVisible: true,
          secondsVisible: false,
        },
        rightPriceScale: { borderColor: '#374151' },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      });
      const ceOiSeries = chartOi.addLineSeries({ color: '#3b82f6', lineWidth: 2 });
      const peOiSeries = chartOi.addLineSeries({ color: '#f97316', lineWidth: 2 });
      const oiZeroSeries = chartOi.addLineSeries({
        color: '#ffffff',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
      });

      const chartVolContainer = document.getElementById('chart-vol');
      const chartVol = LightweightCharts.createChart(chartVolContainer, {
        layout: { background: { color: '#0b1622' }, textColor: '#d1d5db' },
        grid: {
          vertLines: { color: '#1f2933' },
          horzLines: { color: '#1f2933' },
        },
        timeScale: {
          borderColor: '#374151',
          timeVisible: true,
          secondsVisible: false,
        },
        rightPriceScale: { borderColor: '#374151' },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      });
      const ceVolSeries = chartVol.addLineSeries({ color: '#22c55e', lineWidth: 2 });
      const peVolSeries = chartVol.addLineSeries({ color: '#ef4444', lineWidth: 2 });
      const volZeroSeries = chartVol.addLineSeries({
        color: '#ffffff',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
      });

      // --- Sync zoom/pan between charts (time scale) ---
      let isSyncing = false;
      function isValidRange(range) {
        return !!(range && range.from != null && range.to != null);
      }
      function syncTimeScale(sourceChart, targetChart) {
        if (isSyncing) return;
        const range = sourceChart.timeScale().getVisibleRange();
        if (!isValidRange(range)) return;
        isSyncing = true;
        try {
          // setVisibleRange throws if range values are null
          targetChart.timeScale().setVisibleRange(range);
        } catch (e) {
          // ignore transient range errors during initial render / empty datasets
          // console.debug('syncTimeScale ignored:', e);
        } finally {
          isSyncing = false;
        }
      }

      chartOi.timeScale().subscribeVisibleTimeRangeChange(() => {
        syncTimeScale(chartOi, chartVol);
      });
      chartVol.timeScale().subscribeVisibleTimeRangeChange(() => {
        syncTimeScale(chartVol, chartOi);
      });

      // Also keep the candle chart in sync
      chart1m.timeScale().subscribeVisibleTimeRangeChange(() => {
        syncTimeScale(chart1m, chartOi);
        syncTimeScale(chart1m, chartVol);
      });
      chartOi.timeScale().subscribeVisibleTimeRangeChange(() => {
        syncTimeScale(chartOi, chart1m);
      });

      function setStatus(msg) {
        statusEl.textContent = msg || '';
      }

      async function loadData() {
        const ex = exchangeSelect.value || 'NSE';
        const sym = (symbolSelect.value || '').trim();
        const start = startInput.value;
        const end = endInput.value;
        if (!start || !end) {
          setStatus('Please select both start and end dates.');
          return;
        }
        loadBtn.disabled = true;
        setStatus('Loading...');
        try {
          // 1) Load candles
          {
            const params = new URLSearchParams({ exchange: ex, start, end });
            if (sym) params.set('symbol', sym);
            const resp = await fetch('/api/bars_1m?' + params.toString());
            if (!resp.ok) throw new Error('Bars HTTP ' + resp.status);
            const data = await resp.json();
            const bars = data.bars || [];
            // Filter out null OHLC
            const candleData = bars
              .filter(b => b.open != null && b.high != null && b.low != null && b.close != null)
              .map(b => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close }));
            candleSeries.setData(candleData);
            if (data.symbol && !sym) {
              symbolSelect.value = data.symbol; // show auto-selected symbol
            }
          }

          // 2) Load actual trade markers from trade_logs only
          const markers = [];
          {
            const params = new URLSearchParams({ start, end });
            if (ex) params.set('exchange', ex);
            const sym2 = (symbolSelect.value || '').trim();
            if (sym2) params.set('symbol', sym2);
            const resp = await fetch('/api/trade_logs?' + params.toString());
            if (resp.ok) {
              const data = await resp.json();
              const trades = data.trades || [];
              for (const t of trades) {
                if (t.entry_time) {
                  markers.push({
                    time: t.entry_time,
                    position: 'belowBar',
                    color: '#60a5fa',
                    shape: t.side === 'SELL' ? 'arrowDown' : 'arrowUp',
                    text: `${t.side} @ ${t.entry_price || ''}`,
                  });
                }
                if (t.exit_time) {
                  markers.push({
                    time: t.exit_time,
                    position: 'aboveBar',
                    color: (t.pnl != null && t.pnl < 0) ? '#ef4444' : '#22c55e',
                    shape: 'circle',
                    text: `EXIT ${t.pnl != null ? (t.pnl > 0 ? '+' : '') + t.pnl.toFixed(0) : ''}`,
                  });
                }
              }
            }
          }

          // Set markers on candlestick chart (only actual trades)
          candleSeries.setMarkers(markers);

          const params = new URLSearchParams({ exchange: ex, start, end });
          const resp = await fetch('/api/itm_oi_volume?' + params.toString());
          if (!resp.ok) {
            throw new Error('HTTP ' + resp.status);
          }
          const data = await resp.json();
          const points = data.points || [];
          if (!points.length) {
            ceOiSeries.setData([]);
            peOiSeries.setData([]);
            ceVolSeries.setData([]);
            peVolSeries.setData([]);
            setStatus('No data for this range.');
            return;
          }

          const ceOiData = [];
          const peOiData = [];
          const ceVolData = [];
          const peVolData = [];

          for (const p of points) {
            if (p.ce_oi_pct != null) ceOiData.push({ time: p.time, value: p.ce_oi_pct });
            if (p.pe_oi_pct != null) peOiData.push({ time: p.time, value: p.pe_oi_pct });
            if (p.ce_vol_pct != null) ceVolData.push({ time: p.time, value: p.ce_vol_pct });
            if (p.pe_vol_pct != null) peVolData.push({ time: p.time, value: p.pe_vol_pct });
          }

          // Horizontal zero line on both charts using first/last time
          const firstTime = points[0].time;
          const lastTime = points[points.length - 1].time;
          const zeroLineData = [
            { time: firstTime, value: 0 },
            { time: lastTime, value: 0 },
          ];
          oiZeroSeries.setData(zeroLineData);
          volZeroSeries.setData(zeroLineData);

          ceOiSeries.setData(ceOiData);
          peOiSeries.setData(peOiData);
          ceVolSeries.setData(ceVolData);
          peVolSeries.setData(peVolData);

          setStatus(`Loaded ${points.length} points for ${data.exchange} from ${data.start} to ${data.end}.`);
        } catch (err) {
          console.error(err);
          setStatus('Error loading data: ' + err.message);
        } finally {
          loadBtn.disabled = false;
        }
      }

      loadBtn.addEventListener('click', loadData);

      // Initial load
      window.addEventListener('load', loadData);
    </script>
  </body>
  </html>
    """
    return html


if __name__ == "__main__":
    # Run a small standalone server on port 5055
    app.run(host="127.0.0.1", port=5055, debug=False)

