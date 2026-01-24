"""
FastAPI Dashboard Server for OI Gemini (Phase 2 Advanced Visualization).
Provides real-time WebSockets and REST API for the frontend dashboard.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from datetime import datetime, timedelta

# If we need access to AppManager, we'll need a way to share state.
# Since AppManager runs in a different thread/process context usually,
# we might use a singleton or pass it during startup if running in same process.
# Here we assume it's running in the same process but different thread.

app = FastAPI()
LOGGER = logging.getLogger("DashboardServer")

# Global reference to AppManager (will be injected)
_APP_MANAGER = None

def set_app_manager(manager):
    global _APP_MANAGER
    _APP_MANAGER = manager

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass # Handle disconnects gracefully

manager = ConnectionManager()

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root dashboard page with system overview."""
    if _APP_MANAGER:
        exchanges = list(_APP_MANAGER.exchange_handlers.keys())
        exchange_data = {}
        for ex_name, handler in _APP_MANAGER.exchange_handlers.items():
            exchange_data[ex_name] = {
                "spot": handler.latest_oi_data.get('underlying_price'),
                "future": handler.latest_future_price,
                "atm": handler.latest_oi_data.get('atm_strike'),
                "ml_signal": handler.ml_signal if hasattr(handler, 'ml_signal') else 'N/A',
                "ml_confidence": f"{handler.ml_confidence:.1%}" if hasattr(handler, 'ml_confidence') else 'N/A',
            }
        status = "online"
    else:
        exchanges = []
        exchange_data = {}
        status = "offline"
    
    # Build exchange rows
    exchange_rows = ""
    for ex, data in exchange_data.items():
        spot = f"{data['spot']:,.2f}" if data['spot'] else "N/A"
        future = f"{data['future']:,.2f}" if data['future'] else "N/A"
        atm = f"{data['atm']:,.0f}" if data['atm'] else "N/A"
        signal = data.get('ml_signal', 'N/A')
        confidence = data.get('ml_confidence', 'N/A')
        signal_color = "#4ade80" if signal == "BUY" else "#f87171" if signal == "SELL" else "#94a3b8"
        exchange_rows += f"""
        <tr>
            <td style="font-weight: 600;">{ex}</td>
            <td>{spot}</td>
            <td>{future}</td>
            <td>{atm}</td>
            <td style="color: {signal_color}; font-weight: 600;">{signal}</td>
            <td>{confidence}</td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OI Gemini Dashboard</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
                color: #e4e4e7;
                min-height: 100vh;
                padding: 2rem;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #3f3f5a;
            }}
            h1 {{
                font-size: 2rem;
                background: linear-gradient(90deg, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .status {{
                padding: 0.5rem 1rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            }}
            .status.online {{ background: #166534; color: #4ade80; }}
            .status.offline {{ background: #991b1b; color: #fca5a5; }}
            .card {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 1rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }}
            .card h2 {{
                font-size: 1.25rem;
                margin-bottom: 1rem;
                color: #a5b4fc;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 0.75rem 1rem;
                text-align: left;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            th {{
                color: #94a3b8;
                font-weight: 500;
                font-size: 0.875rem;
                text-transform: uppercase;
            }}
            .api-list {{ list-style: none; }}
            .api-list li {{
                padding: 0.5rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }}
            .api-list a {{
                color: #818cf8;
                text-decoration: none;
            }}
            .api-list a:hover {{ text-decoration: underline; }}
            .api-list code {{
                background: rgba(0,0,0,0.3);
                padding: 0.2rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
            }}
            .timestamp {{
                color: #64748b;
                font-size: 0.875rem;
            }}
            button:hover {{
                opacity: 0.9;
                transform: translateY(-1px);
            }}
            button:active {{
                transform: translateY(0);
            }}
            select, input[type="date"] {{
                cursor: pointer;
            }}
            select:hover, input[type="date"]:hover {{
                border-color: rgba(255,255,255,0.4);
            }}
            input[type="date"]::-webkit-calendar-picker-indicator {{
                filter: invert(1);
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔮 OI Gemini Dashboard</h1>
                <span class="status {status}">{status.upper()}</span>
            </div>
            
            <div class="card" style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                    <h2 style="margin: 0; color: #a5b4fc;">📅 Date Range</h2>
                    <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
                        <label style="color: #94a3b8; font-size: 0.875rem;">
                            Start Date:
                            <input type="date" id="backtest-start-date" style="margin-left: 0.5rem; padding: 0.25rem 0.5rem; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 0.25rem; color: #e4e4e7;">
                        </label>
                        <label style="color: #94a3b8; font-size: 0.875rem;">
                            End Date:
                            <input type="date" id="backtest-end-date" style="margin-left: 0.5rem; padding: 0.25rem 0.5rem; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 0.25rem; color: #e4e4e7;">
                        </label>
                        <button onclick="refreshAllData()" style="padding: 0.5rem 1rem; background: #10b981; border: none; border-radius: 0.5rem; color: white; cursor: pointer; font-weight: 600; font-size: 0.875rem;">
                            🔄 Refresh All
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Exchange Status</h2>
                <div id="exchange-status">
                <table>
                    <thead>
                        <tr>
                            <th>Exchange</th>
                            <th>Spot Price</th>
                            <th>Future Price</th>
                            <th>ATM Strike</th>
                            <th>ML Signal</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {exchange_rows if exchange_rows else '<tr><td colspan="6" style="text-align: center; color: #94a3b8;">No exchange data available</td></tr>'}
                    </tbody>
                </table>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Backtest Results</h2>
                <div id="backtest-results" style="color: #94a3b8;">Loading...</div>
            </div>
            
            <div class="card">
                <h2>🤖 Model Performance</h2>
                <div id="model-performance" style="color: #94a3b8;">Loading...</div>
            </div>
            
            <div class="card">
                <h2>📊 Dashboards</h2>
                <ul class="api-list">
                    <li><a href="/oi-change-analysis"><code>OI Change Analysis</code></a> - Visualize OI change metrics across all exchanges</li>
                </ul>
            </div>
            
            <div class="card">
                <h2>🔌 API Endpoints</h2>
                <ul class="api-list">
                    <li><a href="/api/status"><code>GET /api/status</code></a> - System status and available exchanges</li>
                    <li><a href="/api/state/NSE"><code>GET /api/state/{{exchange}}</code></a> - Get state for specific exchange (e.g., NSE, BSE)</li>
                    <li><a href="/api/backtest-results?exchange=NSE"><code>GET /api/backtest-results</code></a> - Backtest performance metrics</li>
                    <li><a href="/api/model-performance?exchange=NSE"><code>GET /api/model-performance</code></a> - Model training metrics</li>
                    <li><code>WS /ws/live</code> - WebSocket endpoint for real-time updates</li>
                </ul>
            </div>
            
            <script>
                // Initialize date fields with defaults (last 7 days)
                function initializeDates() {{
                    const endDate = new Date();
                    const startDate = new Date();
                    startDate.setDate(startDate.getDate() - 7);
                    
                    document.getElementById('backtest-end-date').valueAsDate = endDate;
                    document.getElementById('backtest-start-date').valueAsDate = startDate;
                }}
                
                // Refresh all data sections
                function refreshAllData() {{
                    loadExchangeStatus();
                    loadBacktestResults();
                    loadModelPerformance();
                }}
                
                // Fetch and display exchange status
                async function loadExchangeStatus() {{
                    try {{
                        const startDate = document.getElementById('backtest-start-date').value;
                        const endDate = document.getElementById('backtest-end-date').value;
                        
                        let days = 7;
                        if (startDate && endDate) {{
                            const start = new Date(startDate);
                            const end = new Date(endDate);
                            days = Math.ceil((end - start) / (1000 * 60 * 60 * 24)) + 1;
                        }}
                        
                        // Fetch data for both NSE and BSE
                        const [nseResponse, bseResponse] = await Promise.all([
                            fetch('/api/exchange-status?exchange=NSE&days=' + days),
                            fetch('/api/exchange-status?exchange=BSE&days=' + days)
                        ]);
                        
                        const nseData = await nseResponse.json();
                        const bseData = await bseResponse.json();
                        
                        let html = '<table><thead><tr><th>Exchange</th><th>Spot Price</th><th>Future Price</th><th>PCR</th><th>Signal</th><th>Timestamp</th></tr></thead><tbody>';
                        
                        // Add NSE row
                        if (nseData.success && nseData.spot_price) {{
                            const signalColor = nseData.signal === 'BULLISH' ? '#4ade80' : nseData.signal === 'BEARISH' ? '#f87171' : '#94a3b8';
                            const timestamp = nseData.timestamp ? new Date(nseData.timestamp).toLocaleString() : 'N/A';
                            html += '<tr>';
                            html += '<td style="font-weight: 600;">NSE</td>';
                            html += '<td>' + (nseData.spot_price ? '₹' + nseData.spot_price.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) : 'N/A') + '</td>';
                            html += '<td>' + (nseData.future_price ? '₹' + nseData.future_price.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) : 'N/A') + '</td>';
                            html += '<td>' + (nseData.pcr ? nseData.pcr.toFixed(2) : 'N/A') + '</td>';
                            html += '<td style="color: ' + signalColor + '; font-weight: 600;">' + (nseData.signal || 'N/A') + '</td>';
                            html += '<td style="font-size: 0.875rem; color: #94a3b8;">' + timestamp + '</td>';
                            html += '</tr>';
                        }}
                        
                        // Add BSE row
                        if (bseData.success && bseData.spot_price) {{
                            const signalColor = bseData.signal === 'BULLISH' ? '#4ade80' : bseData.signal === 'BEARISH' ? '#f87171' : '#94a3b8';
                            const timestamp = bseData.timestamp ? new Date(bseData.timestamp).toLocaleString() : 'N/A';
                            html += '<tr>';
                            html += '<td style="font-weight: 600;">BSE</td>';
                            html += '<td>' + (bseData.spot_price ? '₹' + bseData.spot_price.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) : 'N/A') + '</td>';
                            html += '<td>' + (bseData.future_price ? '₹' + bseData.future_price.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) : 'N/A') + '</td>';
                            html += '<td>' + (bseData.pcr ? bseData.pcr.toFixed(2) : 'N/A') + '</td>';
                            html += '<td style="color: ' + signalColor + '; font-weight: 600;">' + (bseData.signal || 'N/A') + '</td>';
                            html += '<td style="font-size: 0.875rem; color: #94a3b8;">' + timestamp + '</td>';
                            html += '</tr>';
                        }}
                        
                        html += '</tbody></table>';
                        
                        if (!nseData.success && !bseData.success) {{
                            html = '<div style="color: #94a3b8; padding: 1rem;">No exchange data available for selected date range</div>';
                        }}
                        
                        document.getElementById('exchange-status').innerHTML = html;
                    }} catch (error) {{
                        document.getElementById('exchange-status').innerHTML = '<div style="color: #f87171; padding: 1rem;">Error loading exchange status</div>';
                    }}
                }}
                
                // Fetch and display backtest results
                async function loadBacktestResults() {{
                    try {{
                        const startDate = document.getElementById('backtest-start-date').value;
                        const endDate = document.getElementById('backtest-end-date').value;
                        
                        let days = 7;
                        if (startDate && endDate) {{
                            const start = new Date(startDate);
                            const end = new Date(endDate);
                            days = Math.ceil((end - start) / (1000 * 60 * 60 * 24)) + 1;
                        }}
                        
                        // Fetch data for both NSE and BSE
                        const [nseResponse, bseResponse] = await Promise.all([
                            fetch('/api/backtest-results?exchange=NSE&days=' + days),
                            fetch('/api/backtest-results?exchange=BSE&days=' + days)
                        ]);
                        
                        const nseData = await nseResponse.json();
                        const bseData = await bseResponse.json();
                        
                        let html = '';
                        let hasData = false;
                        
                        // NSE Section
                        if (nseData.success && nseData.summary.total_trades > 0) {{
                            hasData = true;
                            const summary = nseData.summary;
                            const pnlColor = summary.total_pnl >= 0 ? '#4ade80' : '#f87171';
                            
                            html += '<h3 style="color: #a5b4fc; margin-bottom: 1rem;">NSE</h3>';
                            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Total Trades</div><div style="font-size: 1.5rem; font-weight: 600;">' + summary.total_trades + '</div></div>';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Total PnL</div><div style="font-size: 1.5rem; font-weight: 600; color: ' + pnlColor + ';">₹' + summary.total_pnl.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) + '</div></div>';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Win Rate</div><div style="font-size: 1.5rem; font-weight: 600;">' + summary.win_rate + '%</div></div>';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Profit Factor</div><div style="font-size: 1.5rem; font-weight: 600;">' + summary.profit_factor + '</div></div>';
                            html += '</div>';
                            
                            if (nseData.by_horizon && nseData.by_horizon.length > 0) {{
                                html += '<table style="margin-top: 1rem;"><thead><tr><th>Horizon</th><th>Trades</th><th>Total PnL</th><th>Avg PnL</th><th>Win Rate</th></tr></thead><tbody>';
                                nseData.by_horizon.forEach(h => {{
                                    const pnlColor = h.total_pnl >= 0 ? '#4ade80' : '#f87171';
                                    html += '<tr><td>' + h.horizon + '</td><td>' + h.trades + '</td>';
                                    html += '<td style="color: ' + pnlColor + ';">₹' + h.total_pnl.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) + '</td>';
                                    html += '<td>₹' + h.avg_pnl.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) + '</td>';
                                    html += '<td>' + h.win_rate + '%</td></tr>';
                                }});
                                html += '</tbody></table>';
                            }}
                        }}
                        
                        // BSE Section
                        if (bseData.success && bseData.summary.total_trades > 0) {{
                            hasData = true;
                            const summary = bseData.summary;
                            const pnlColor = summary.total_pnl >= 0 ? '#4ade80' : '#f87171';
                            
                            html += '<h3 style="color: #a5b4fc; margin: 2rem 0 1rem 0;">BSE</h3>';
                            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Total Trades</div><div style="font-size: 1.5rem; font-weight: 600;">' + summary.total_trades + '</div></div>';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Total PnL</div><div style="font-size: 1.5rem; font-weight: 600; color: ' + pnlColor + ';">₹' + summary.total_pnl.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) + '</div></div>';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Win Rate</div><div style="font-size: 1.5rem; font-weight: 600;">' + summary.win_rate + '%</div></div>';
                            html += '<div><div style="color: #94a3b8; font-size: 0.875rem;">Profit Factor</div><div style="font-size: 1.5rem; font-weight: 600;">' + summary.profit_factor + '</div></div>';
                            html += '</div>';
                            
                            if (bseData.by_horizon && bseData.by_horizon.length > 0) {{
                                html += '<table style="margin-top: 1rem;"><thead><tr><th>Horizon</th><th>Trades</th><th>Total PnL</th><th>Avg PnL</th><th>Win Rate</th></tr></thead><tbody>';
                                bseData.by_horizon.forEach(h => {{
                                    const pnlColor = h.total_pnl >= 0 ? '#4ade80' : '#f87171';
                                    html += '<tr><td>' + h.horizon + '</td><td>' + h.trades + '</td>';
                                    html += '<td style="color: ' + pnlColor + ';">₹' + h.total_pnl.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) + '</td>';
                                    html += '<td>₹' + h.avg_pnl.toLocaleString('en-IN', {{maximumFractionDigits: 2}}) + '</td>';
                                    html += '<td>' + h.win_rate + '%</td></tr>';
                                }});
                                html += '</tbody></table>';
                            }}
                        }}
                        
                        if (!hasData) {{
                            html = '<div style="color: #94a3b8;">No backtest data available for selected date range</div>';
                        }}
                        
                        document.getElementById('backtest-results').innerHTML = html;
                    }} catch (error) {{
                        document.getElementById('backtest-results').innerHTML = '<div style="color: #f87171;">Error loading backtest results</div>';
                    }}
                }}
                
                // Fetch and display model performance
                async function loadModelPerformance() {{
                    try {{
                        const response = await fetch('/api/model-performance?exchange=NSE');
                        const data = await response.json();
                        
                        if (data.success && data.models) {{
                            let html = '';
                            
                            if (data.training_start_date && data.training_end_date) {{
                                const startDate = data.training_start_date.split('T')[0];
                                const endDate = data.training_end_date.split('T')[0];
                                html += '<div style="color: #94a3b8; margin-bottom: 1rem; font-size: 0.875rem;">';
                                html += 'Training Period: ' + startDate + ' to ' + endDate;
                                html += ' (' + data.training_duration_days + ' days, ' + data.n_samples + ' samples)';
                                html += '</div>';
                            }}
                            
                            html += '<table><thead><tr><th>Model</th><th>R²</th><th>MAE</th><th>Direction Accuracy</th><th>Return on Long</th></tr></thead><tbody>';
                            
                            Object.entries(data.models).forEach(([model, metrics]) => {{
                                const accColor = metrics.direction_accuracy >= 60 ? '#4ade80' : metrics.direction_accuracy >= 50 ? '#fbbf24' : '#f87171';
                                html += '<tr><td>' + model + '</td><td>' + metrics.r2 + '</td>';
                                html += '<td>' + metrics.mae.toFixed(2) + '</td>';
                                html += '<td style="color: ' + accColor + ';">' + metrics.direction_accuracy + '%</td>';
                                html += '<td>' + metrics.return_on_long.toFixed(2) + '%</td></tr>';
                            }});
                            
                            html += '</tbody></table>';
                            document.getElementById('model-performance').innerHTML = html;
                        }} else {{
                            document.getElementById('model-performance').innerHTML = '<div style="color: #94a3b8;">No model performance data available</div>';
                        }}
                    }} catch (error) {{
                        document.getElementById('model-performance').innerHTML = '<div style="color: #f87171;">Error loading model performance</div>';
                    }}
                }}
                
                // Initialize dates and load data on page load
                initializeDates();
                loadExchangeStatus();
                loadBacktestResults();
                loadModelPerformance();
            </script>
            
            <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (use refresh buttons to update data)</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/status")
async def get_status():
    if _APP_MANAGER:
        return {"status": "online", "exchanges": list(_APP_MANAGER.exchange_handlers.keys())}
    return {"status": "offline"}

@app.get("/api/state/{exchange}")
async def get_exchange_state(exchange: str):
    if not _APP_MANAGER:
        return {"error": "AppManager not initialized"}
    
    handler = _APP_MANAGER.exchange_handlers.get(exchange)
    if not handler:
        return {"error": "Exchange not found"}
        
    # Snapshot of critical state - use latest_oi_data for spot price and ATM
    return {
        "spot_price": handler.latest_oi_data.get('underlying_price'),
        "future_price": handler.latest_future_price,
        "expiry": handler.expiry_date.isoformat() if handler.expiry_date else None,
        "positions": _APP_MANAGER.open_positions if hasattr(_APP_MANAGER, 'open_positions') else {}
    }

@app.get("/api/exchange-status")
async def get_exchange_status(
    exchange: str = Query("NSE", description="Exchange name (e.g., NSE, BSE)"),
    days: int = Query(7, description="Number of days to look back")
):
    """
    Fetch exchange status data from ml_features table for the specified date range.
    Returns latest prices and key metrics within the date range.
    """
    try:
        import database_new as db
        from config import get_config
        
        config = get_config()
        ph = '%s' if config.db_type == 'postgres' else '?'
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = None
        try:
            conn = db.get_db_connection()
            cursor = conn.cursor()
            
            # Get latest data within date range
            query = f"""
                SELECT 
                    exchange,
                    underlying_price,
                    underlying_future_price,
                    pcr_total_oi,
                    timestamp
                FROM ml_features
                WHERE exchange = {ph}
                  AND timestamp >= {ph}
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            cursor.execute(query, (exchange, cutoff_date))
            row = cursor.fetchone()
            
            if row:
                exchange_name, spot_price, future_price, pcr, timestamp = row
                
                # Determine signal based on PCR (simple logic)
                signal = "N/A"
                if pcr is not None:
                    if pcr > 1.2:
                        signal = "BULLISH"
                    elif pcr < 0.8:
                        signal = "BEARISH"
                    else:
                        signal = "NEUTRAL"
                
                return {
                    "success": True,
                    "exchange": exchange_name,
                    "spot_price": float(spot_price) if spot_price else None,
                    "future_price": float(future_price) if future_price else None,
                    "pcr": float(pcr) if pcr else None,
                    "timestamp": timestamp.isoformat() if timestamp else None,
                    "signal": signal
                }
            else:
                return {
                    "success": False,
                    "error": f"No data found for {exchange} in the last {days} days"
                }
                
        except Exception as e:
            LOGGER.error(f"Error fetching exchange status: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if conn:
                db.release_db_connection(conn)
                
    except Exception as e:
        LOGGER.error(f"Error in get_exchange_status: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/api/backtest-results")
async def get_backtest_results(
    exchange: str = Query("NSE", description="Exchange name (e.g., NSE, BSE)"),
    days: int = Query(7, description="Number of days to look back")
):
    """
    Fetch backtest results from option_return_backtest_trades table.
    Returns summary metrics including total PnL, win rate, and performance by horizon.
    """
    try:
        import database_new as db
        from config import get_config
        
        config = get_config()
        ph = '%s' if config.db_type == 'postgres' else '?'
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = None
        try:
            conn = db.get_db_connection()
            cursor = conn.cursor()
            
            # Overall summary
            summary_query = f"""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(net_pnl) as total_pnl,
                    AVG(net_pnl) as avg_pnl,
                    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN net_pnl < 0 THEN 1 END) as losing_trades,
                    AVG(CASE WHEN net_pnl > 0 THEN net_pnl END) as avg_win,
                    AVG(CASE WHEN net_pnl < 0 THEN net_pnl END) as avg_loss,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM option_return_backtest_trades
                WHERE exchange = {ph}
                  AND timestamp >= {ph}
            """
            
            cursor.execute(summary_query, (exchange, cutoff_date))
            summary_row = cursor.fetchone()
            
            if summary_row and summary_row[0]:
                total_trades, total_pnl, avg_pnl, wins, losses, avg_win, avg_loss, first_trade, last_trade = summary_row
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss and avg_loss != 0 else 0
                
                summary = {
                    "total_trades": int(total_trades) if total_trades else 0,
                    "total_pnl": float(total_pnl) if total_pnl else 0.0,
                    "avg_pnl": float(avg_pnl) if avg_pnl else 0.0,
                    "win_rate": round(win_rate, 2),
                    "winning_trades": int(wins) if wins else 0,
                    "losing_trades": int(losses) if losses else 0,
                    "avg_win": float(avg_win) if avg_win else 0.0,
                    "avg_loss": float(avg_loss) if avg_loss else 0.0,
                    "profit_factor": round(profit_factor, 2) if profit_factor else 0.0,
                    "first_trade": first_trade.isoformat() if first_trade else None,
                    "last_trade": last_trade.isoformat() if last_trade else None
                }
            else:
                summary = {
                    "total_trades": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "win_rate": 0.0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                    "first_trade": None,
                    "last_trade": None
                }
            
            # Performance by horizon
            horizon_query = f"""
                SELECT 
                    horizon,
                    COUNT(*) as trades,
                    SUM(net_pnl) as total_pnl,
                    AVG(net_pnl) as avg_pnl,
                    COUNT(CASE WHEN net_pnl > 0 THEN 1 END) as wins,
                    COUNT(CASE WHEN net_pnl < 0 THEN 1 END) as losses
                FROM option_return_backtest_trades
                WHERE exchange = {ph}
                  AND timestamp >= {ph}
                GROUP BY horizon
                ORDER BY horizon
            """
            
            cursor.execute(horizon_query, (exchange, cutoff_date))
            horizon_rows = cursor.fetchall()
            
            by_horizon = []
            for row in horizon_rows:
                horizon, trades, total_pnl, avg_pnl, wins, losses = row
                win_rate = (wins / trades * 100) if trades > 0 else 0
                by_horizon.append({
                    "horizon": horizon,
                    "trades": int(trades),
                    "total_pnl": float(total_pnl) if total_pnl else 0.0,
                    "avg_pnl": float(avg_pnl) if avg_pnl else 0.0,
                    "win_rate": round(win_rate, 2),
                    "wins": int(wins),
                    "losses": int(losses)
                })
            
            # Performance by option type
            option_type_query = f"""
                SELECT 
                    option_type,
                    COUNT(*) as trades,
                    SUM(net_pnl) as total_pnl,
                    AVG(net_pnl) as avg_pnl
                FROM option_return_backtest_trades
                WHERE exchange = {ph}
                  AND timestamp >= {ph}
                GROUP BY option_type
            """
            
            cursor.execute(option_type_query, (exchange, cutoff_date))
            option_type_rows = cursor.fetchall()
            
            by_option_type = {}
            for row in option_type_rows:
                opt_type, trades, total_pnl, avg_pnl = row
                by_option_type[opt_type] = {
                    "trades": int(trades),
                    "total_pnl": float(total_pnl) if total_pnl else 0.0,
                    "avg_pnl": float(avg_pnl) if avg_pnl else 0.0
                }
            
            return {
                "success": True,
                "exchange": exchange,
                "days": days,
                "summary": summary,
                "by_horizon": by_horizon,
                "by_option_type": by_option_type
            }
            
        except Exception as e:
            LOGGER.error(f"Error fetching backtest results: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if conn:
                db.release_db_connection(conn)
                
    except Exception as e:
        LOGGER.error(f"Error in get_backtest_results: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/api/model-performance")
async def get_model_performance(
    exchange: str = Query("NSE", description="Exchange name (e.g., NSE, BSE)")
):
    """
    Fetch model performance metrics from training summary files.
    Returns metrics for option return models including R², MAE, and direction accuracy.
    """
    try:
        import json
        from pathlib import Path
        
        model_dir = Path(f"models/option_returns/{exchange}")
        summary_path = model_dir / "training_summary.json"
        
        if not summary_path.exists():
            return {
                "success": False,
                "error": f"Training summary not found for {exchange}",
                "exchange": exchange
            }
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Extract model metrics
        model_metrics = summary.get('model_metrics', {})
        
        # Format metrics for display
        formatted_metrics = {}
        for model_name, metrics in model_metrics.items():
            formatted_metrics[model_name] = {
                "r2": round(metrics.get('avg_r2', 0), 4),
                "mae": round(metrics.get('avg_mae', 0), 4),
                "direction_accuracy": round(metrics.get('avg_direction_accuracy', 0) * 100, 2),
                "return_on_long": round(metrics.get('avg_return_on_long', 0), 4)
            }
        
        return {
            "success": True,
            "exchange": exchange,
            "training_date": summary.get('training_date'),
            "training_start_date": summary.get('training_start_date'),
            "training_end_date": summary.get('training_end_date'),
            "training_duration_days": summary.get('training_duration_days'),
            "n_samples": summary.get('n_samples'),
            "models": formatted_metrics
        }
        
    except Exception as e:
        LOGGER.error(f"Error in get_model_performance: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.get("/api/oi-change-data")
async def get_oi_change_data(
    exchange: str = Query("NSE", description="Exchange name (e.g., NSE, BSE)"),
    hours: int = Query(None, description="Number of hours of historical data to fetch"),
    start_date: str = Query(None, description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date in ISO format (YYYY-MM-DD)"),
    all_exchanges: bool = Query(False, description="Fetch data for all exchanges")
):
    """
    Fetch OI change data from ml_features table for visualization.
    Returns time series data for nse_next_oi_change_diff_put_call, 
    nse_next_oi_change_call_total, and nse_next_oi_change_put_total.
    
    Either provide hours OR start_date/end_date. If both are provided, date range takes precedence.
    """
    try:
        # Import database module
        import database_new as db
        from config import get_config
        
        config = get_config()
        ph = '%s' if config.db_type == 'postgres' else '?'
        
        # Calculate time range
        if start_date and end_date:
            # Use date range
            try:
                from time_utils import to_ist
                # Parse dates and set to start/end of day in IST
                start_time = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0)
                end_time = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
                # Convert to IST if needed (database stores in IST)
                start_time = to_ist(start_time)
                end_time = to_ist(end_time)
            except ValueError as e:
                return {"success": False, "error": f"Invalid date format. Use YYYY-MM-DD: {e}"}
        elif hours is not None:
            # Use hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
        else:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
        
        # Get exchanges to query
        if all_exchanges:
            # Get all exchanges from config
            exchanges = config.all_exchanges
        else:
            exchanges = [exchange]
        
        result = {}
        
        for ex in exchanges:
            conn = None
            try:
                conn = db.get_db_connection()
                cursor = conn.cursor()
                
                query = f"""
                    SELECT 
                        timestamp,
                        nse_next_oi_change_diff_put_call,
                        nse_next_oi_change_call_total,
                        nse_next_oi_change_put_total
                    FROM ml_features
                    WHERE exchange = {ph}
                      AND timestamp >= {ph}
                      AND timestamp <= {ph}
                      AND (nse_next_oi_change_diff_put_call IS NOT NULL
                           OR nse_next_oi_change_call_total IS NOT NULL
                           OR nse_next_oi_change_put_total IS NOT NULL)
                    ORDER BY timestamp ASC
                """
                
                cursor.execute(query, (ex, start_time, end_time))
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                data = []
                for row in rows:
                    data.append({
                        "timestamp": row[0].isoformat() if isinstance(row[0], datetime) else str(row[0]),
                        "diff_put_call": float(row[1]) if row[1] is not None else None,
                        "call_total": float(row[2]) if row[2] is not None else None,
                        "put_total": float(row[3]) if row[3] is not None else None,
                    })
                
                result[ex] = data
                
            except Exception as e:
                LOGGER.error(f"Error fetching OI change data for {ex}: {e}", exc_info=True)
                result[ex] = []
            finally:
                if conn:
                    db.release_db_connection(conn)
        
        return {
            "success": True,
            "data": result,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
    except Exception as e:
        LOGGER.error(f"Error in get_oi_change_data: {e}", exc_info=True)
        return {"success": False, "error": str(e), "data": {}}

@app.get("/oi-change-analysis", response_class=HTMLResponse)
async def oi_change_analysis_page():
    """Serve the OI Change Analysis visualization page."""
    try:
        with open("templates/oi_change_analysis.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error</h1><p>OI Change Analysis template not found.</p>",
            status_code=404
        )

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open, client handles pings
            # We can also push data here if we want per-client loops
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Background Task for Broadcasting Updates ---

async def broadcast_state():
    """
    Polls AppManager state and broadcasts to WebSocket clients every second.
    """
    while True:
        if _APP_MANAGER:
            try:
                # Aggregate state from all exchanges
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "exchanges": {}
                }
                
                for ex_name, handler in _APP_MANAGER.exchange_handlers.items():
                    payload["exchanges"][ex_name] = {
                        "spot": handler.latest_oi_data.get('underlying_price'),
                        "future": handler.latest_future_price,
                        "atm": handler.latest_oi_data.get('atm_strike'),
                        # Add latest signal/sentiment if available
                    }
                
                # Check for sentiment
                if hasattr(_APP_MANAGER, 'last_nifty_sentiment_data'):
                    payload["sentiment"] = _APP_MANAGER.last_nifty_sentiment_data
                
                await manager.broadcast(json.dumps(payload))
                
            except Exception as e:
                LOGGER.error(f"Error broadcasting state: {e}")
        
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_state())

def run_dashboard_server(host="0.0.0.0", port=8000, app_manager=None):
    """
    Helper to run Uvicorn programmatically.
    Call this from a separate thread.
    """
    import uvicorn
    if app_manager:
        set_app_manager(app_manager)
    
    # Run uvicorn
    uvicorn.run(app, host=host, port=port, log_level="error")

