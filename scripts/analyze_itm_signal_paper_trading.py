#!/usr/bin/env python3
"""
Analyze paper_trading_metrics for last N days considering ONLY rows where
metadata has itm_bearish_signal=true OR itm_bullish_signal=true (ITM-only trading).

Computes: total signals, executed trades, PnL, win rate, profit factor, by day and by signal type.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database_new as db
import pandas as pd
import argparse
from datetime import datetime, timedelta


def analyze_itm_signal_outcome(days: int = 3, exchange: str = None):
    """Query paper_trading_metrics filtered by itm_bearish_signal / itm_bullish_signal in metadata."""
    conn = db.get_db_connection()

    try:
        # Filter: only rows where metadata has itm_bearish_signal=true OR itm_bullish_signal=true
        # PostgreSQL JSONB: -> returns json, ->> returns text. Stored as JSON true/false.
        interval_sql = f"INTERVAL '{days} days'"
        base_where = f"""
            timestamp >= NOW() - {interval_sql}
            AND (
                COALESCE((metadata->>'itm_bearish_signal') = 'true', false)
                OR COALESCE((metadata->>'itm_bullish_signal') = 'true', false)
            )
        """
        if exchange:
            base_where += f" AND exchange = '{exchange}'"

        # 1) Raw rows for this filter (to count and aggregate)
        query_rows = f"""
            SELECT
                id,
                timestamp,
                exchange,
                executed,
                reason,
                signal,
                confidence,
                quantity_lots,
                pnl,
                metadata->>'signal_id' as signal_id,
                metadata->>'itm_bearish_signal' as itm_bearish,
                metadata->>'itm_bullish_signal' as itm_bullish
            FROM paper_trading_metrics
            WHERE {base_where}
            ORDER BY timestamp DESC
        """
        df_all = pd.read_sql_query(query_rows, conn)

        if df_all.empty:
            print(f"No paper_trading_metrics rows with itm_bearish_signal or itm_bullish_signal in metadata for the last {days} day(s).")
            if exchange:
                print(f"Exchange filter: {exchange}")
            return

        df_all["trade_date"] = pd.to_datetime(df_all["timestamp"]).dt.date
        # JSON booleans come back as 'true' (lowercase) from PostgreSQL ->> operator
        df_all["signal_type"] = "both"
        bearish_true = (df_all["itm_bearish"] == "true") | (df_all["itm_bearish"] == "True")
        bullish_true = (df_all["itm_bullish"] == "true") | (df_all["itm_bullish"] == "True")
        df_all.loc[bearish_true & ~bullish_true, "signal_type"] = "itm_bearish_signal"
        df_all.loc[bullish_true & ~bearish_true, "signal_type"] = "itm_bullish_signal"
        df_all.loc[bearish_true & bullish_true, "signal_type"] = "both"

        # 2) Daily stats (ITM signals only)
        query_daily = f"""
            SELECT
                DATE(timestamp) as trade_date,
                exchange,
                COUNT(*) as total_signals,
                SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_trades,
                SUM(CASE WHEN executed = FALSE THEN 1 ELSE 0 END) as skipped_trades,
                SUM(CASE WHEN executed AND pnl IS NOT NULL THEN pnl ELSE 0 END) as total_pnl,
                AVG(CASE WHEN executed AND pnl IS NOT NULL THEN pnl ELSE NULL END) as avg_pnl,
                SUM(CASE WHEN executed AND pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN executed AND pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(CASE WHEN executed AND pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN executed AND pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss,
                CASE
                    WHEN SUM(CASE WHEN executed AND pnl < 0 THEN ABS(pnl) ELSE 0 END) > 0
                    THEN SUM(CASE WHEN executed AND pnl > 0 THEN pnl ELSE 0 END)::float /
                         SUM(CASE WHEN executed AND pnl < 0 THEN ABS(pnl) ELSE 0 END)::float
                    ELSE NULL
                END as profit_factor,
                ROUND(
                    (CASE
                        WHEN SUM(CASE WHEN executed THEN 1 ELSE 0 END) > 0
                        THEN 100.0 * SUM(CASE WHEN executed AND pnl > 0 THEN 1 ELSE 0 END)::float /
                             SUM(CASE WHEN executed THEN 1 ELSE 0 END)::float
                        ELSE 0
                    END)::numeric, 2
                ) as win_rate_pct
            FROM paper_trading_metrics
            WHERE {base_where}
            GROUP BY DATE(timestamp), exchange
            ORDER BY trade_date DESC
        """
        df_daily = pd.read_sql_query(query_daily, conn)

        # 3) By signal type (bearish vs bullish) - need to tag rows; we'll aggregate from df_all
        executed = df_all[df_all["executed"] == True].copy()
        executed_with_pnl = executed[executed["pnl"].notna()]

        # Overall stats (ITM-only filter)
        total_signals = len(df_all)
        total_executed = int(df_all["executed"].sum())
        total_skipped = total_signals - total_executed
        total_pnl = float(executed["pnl"].sum()) if executed["pnl"].notna().any() else 0.0
        wins = int((executed["pnl"] > 0).sum()) if "pnl" in executed.columns else 0
        losses = int((executed["pnl"] < 0).sum()) if "pnl" in executed.columns else 0
        gross_profit = float(executed.loc[executed["pnl"] > 0, "pnl"].sum()) if (executed["pnl"] > 0).any() else 0.0
        gross_loss = float(executed.loc[executed["pnl"] < 0, "pnl"].abs().sum()) if (executed["pnl"] < 0).any() else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else float("inf"))
        win_rate_pct = (100.0 * wins / total_executed) if total_executed > 0 else 0.0

        # By signal type from raw data
        bearish = df_all[(df_all["itm_bearish"] == "true") | (df_all["itm_bearish"] == "True")]
        bullish = df_all[(df_all["itm_bullish"] == "true") | (df_all["itm_bullish"] == "True")]
        exec_bearish = bearish[bearish["executed"] == True]
        exec_bullish = bullish[bullish["executed"] == True]

        def _stats(sub):
            if sub.empty:
                return {"n": 0, "executed": 0, "pnl": 0.0, "wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0, "win_rate": 0.0, "pf": None}
            ex = sub[sub["executed"] == True]
            if ex.empty:
                return {"n": len(sub), "executed": 0, "pnl": 0.0, "wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0, "win_rate": 0.0, "pf": None}
            pnl = float(ex["pnl"].sum()) if ex["pnl"].notna().any() else 0.0
            w = int((ex["pnl"] > 0).sum())
            l = int((ex["pnl"] < 0).sum())
            gp = float(ex.loc[ex["pnl"] > 0, "pnl"].sum()) if (ex["pnl"] > 0).any() else 0.0
            gl = float(ex.loc[ex["pnl"] < 0, "pnl"].abs().sum()) if (ex["pnl"] < 0).any() else 0.0
            wr = (100.0 * w / len(ex)) if len(ex) > 0 else 0.0
            pf = (gp / gl) if gl > 0 else (None if gp == 0 else float("inf"))
            return {"n": len(sub), "executed": len(ex), "pnl": pnl, "wins": w, "losses": l, "gross_profit": gp, "gross_loss": gl, "win_rate": wr, "pf": pf}

        st_bearish = _stats(bearish)
        st_bullish = _stats(bullish)

        # 4) Outcome from trade_logs: match by signal_id to get realized PnL for ITM-only trades
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        trade_logs_dir = os.path.join(project_root, "trade_logs")
        itm_signal_ids = set(df_all["signal_id"].dropna().astype(str).unique())
        trade_log_frames = []
        for d in range(days):
            dt = (datetime.now() - timedelta(days=d)).date()
            path = os.path.join(trade_logs_dir, f"trades_{dt}.csv")
            if os.path.isfile(path):
                try:
                    tl = pd.read_csv(path)
                    if "signal_id" in tl.columns:
                        tl["trade_date"] = pd.to_datetime(tl["entry_timestamp"], errors="coerce").dt.date
                        trade_log_frames.append(tl)
                except Exception as e:
                    pass  # skip missing or bad CSV
        if trade_log_frames and itm_signal_ids:
            tl_all = pd.concat(trade_log_frames, ignore_index=True)
            tl_closed = tl_all[tl_all["status"].astype(str).str.upper() == "CLOSED"].copy()
            tl_itm = tl_closed[tl_closed["signal_id"].astype(str).isin(itm_signal_ids)]
            if not tl_itm.empty and "pnl" in tl_itm.columns:
                pnl_series = pd.to_numeric(tl_itm["pnl"], errors="coerce").fillna(0)
                total_pnl_tl = float(pnl_series.sum())
                wins_tl = int((pnl_series > 0).sum())
                losses_tl = int((pnl_series < 0).sum())
                n_trades_tl = len(tl_itm)
                gross_profit_tl = float(pnl_series[pnl_series > 0].sum())
                gross_loss_tl = float(pnl_series[pnl_series < 0].abs().sum())
                win_rate_tl = (100.0 * wins_tl / n_trades_tl) if n_trades_tl else 0.0
                pf_tl = (gross_profit_tl / gross_loss_tl) if gross_loss_tl > 0 else (None if gross_profit_tl == 0 else float("inf"))
                outcome_from_trade_logs = {
                    "n_trades": n_trades_tl,
                    "total_pnl": total_pnl_tl,
                    "wins": wins_tl,
                    "losses": losses_tl,
                    "win_rate": win_rate_tl,
                    "gross_profit": gross_profit_tl,
                    "gross_loss": gross_loss_tl,
                    "profit_factor": pf_tl,
                }
            else:
                outcome_from_trade_logs = None
        else:
            outcome_from_trade_logs = None

        # ----- Print report -----
        print("=" * 80)
        print("ITM-ONLY PAPER TRADING OUTCOME (itm_bearish_signal + itm_bullish_signal)")
        print(f"Last {days} day(s)" + (f" | Exchange: {exchange}" if exchange else " | All exchanges"))
        print("=" * 80)

        print("\n--- Daily breakdown ---\n")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df_daily.to_string(index=False))

        print("\n--- By signal type (metadata) ---\n")
        print("itm_bearish_signal:")
        print(f"  Signals: {st_bearish['n']}, Executed: {st_bearish['executed']}, Total PnL: ₹{st_bearish['pnl']:,.2f}")
        print(f"  Wins: {st_bearish['wins']}, Losses: {st_bearish['losses']}, Win rate: {st_bearish['win_rate']:.1f}%")
        print(f"  Gross profit: ₹{st_bearish['gross_profit']:,.2f}, Gross loss: ₹{st_bearish['gross_loss']:,.2f}, Profit factor: {st_bearish['pf']}")
        print("itm_bullish_signal:")
        print(f"  Signals: {st_bullish['n']}, Executed: {st_bullish['executed']}, Total PnL: ₹{st_bullish['pnl']:,.2f}")
        print(f"  Wins: {st_bullish['wins']}, Losses: {st_bullish['losses']}, Win rate: {st_bullish['win_rate']:.1f}%")
        print(f"  Gross profit: ₹{st_bullish['gross_profit']:,.2f}, Gross loss: ₹{st_bullish['gross_loss']:,.2f}, Profit factor: {st_bullish['pf']}")

        # Count executed rows with non-null PnL (closed trades)
        executed_with_pnl = int(executed["pnl"].notna().sum())
        print("\n--- Overall (ITM-only filter) ---\n")
        print(f"Total signals (itm_bearish or itm_bullish): {total_signals}")
        print(f"Executed: {total_executed}, Skipped: {total_skipped}")
        if total_signals > 0:
            print(f"Execution rate: {100.0 * total_executed / total_signals:.1f}%")
        print(f"Executed with PnL (closed): {executed_with_pnl} (of {total_executed})")
        if executed_with_pnl == 0 and total_executed > 0:
            print("(PnL is filled when position is closed; 0 closed trades in this window.)")
        print(f"Total PnL: ₹{total_pnl:,.2f}")
        print(f"Winning trades: {wins}, Losing trades: {losses}")
        print(f"Win rate: {win_rate_pct:.1f}%")
        print(f"Gross profit: ₹{gross_profit:,.2f}, Gross loss: ₹{gross_loss:,.2f}")
        print(f"Profit factor: {profit_factor}")

        if outcome_from_trade_logs:
            o = outcome_from_trade_logs
            print("\n--- Outcome from trade_logs (closed trades matched by signal_id, ITM-only) ---\n")
            print(f"Closed trades (ITM signal_id): {o['n_trades']}")
            print(f"Total PnL: ₹{o['total_pnl']:,.2f}")
            print(f"Winning: {o['wins']}, Losing: {o['losses']}, Win rate: {o['win_rate']:.1f}%")
            print(f"Gross profit: ₹{o['gross_profit']:,.2f}, Gross loss: ₹{o['gross_loss']:,.2f}, Profit factor: {o['profit_factor']}")

        return {
            "daily": df_daily,
            "overall": {
                "total_signals": total_signals,
                "executed": total_executed,
                "total_pnl": total_pnl,
                "win_rate_pct": win_rate_pct,
                "profit_factor": profit_factor,
                "bearish": st_bearish,
                "bullish": st_bullish,
            },
        }
    finally:
        db.release_db_connection(conn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITM-only paper trading outcome (last N days)")
    parser.add_argument("--days", type=int, default=3, help="Number of days (default: 3)")
    parser.add_argument("--exchange", type=str, help="Filter by exchange (NSE or BSE)")
    args = parser.parse_args()
    analyze_itm_signal_outcome(days=args.days, exchange=args.exchange)
