"""
FastAPI Dashboard Server for OI Gemini (Phase 2 Advanced Visualization).
Provides real-time WebSockets and REST API for the frontend dashboard.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

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
        <meta http-equiv="refresh" content="5">
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔮 OI Gemini Dashboard</h1>
                <span class="status {status}">{status.upper()}</span>
            </div>
            
            <div class="card">
                <h2>📊 Exchange Status</h2>
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
            
            <div class="card">
                <h2>🔌 API Endpoints</h2>
                <ul class="api-list">
                    <li><a href="/api/status"><code>GET /api/status</code></a> - System status and available exchanges</li>
                    <li><a href="/api/state/NSE"><code>GET /api/state/{{exchange}}</code></a> - Get state for specific exchange (e.g., NSE, BSE)</li>
                    <li><code>WS /ws/live</code> - WebSocket endpoint for real-time updates</li>
                </ul>
            </div>
            
            <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (auto-refreshes every 5s)</p>
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

