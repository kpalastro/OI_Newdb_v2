from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate model for an exchange against simple absolute thresholds."
    )
    parser.add_argument(
        "--exchange",
        required=True,
        help="Exchange key (e.g. NSE, BSE).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Optional evaluation tag used in the output filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ex = args.exchange.upper()
    reports_dir = Path("models") / ex / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    training_report = _load_json(reports_dir / "training_report.json")
    auto_ml_summary = _load_json(reports_dir / "auto_ml_summary.json")

    # Backtest: prefer models/<EX>/reports/backtest_summary.json, fallback to reports/backtests/<EX>.json
    backtest_report = _load_json(reports_dir / "backtest_summary.json")
    if not backtest_report:
        fallback_bt = Path("reports") / "backtests" / f"{ex}.json"
        backtest_report = _load_json(fallback_bt)

    # ---- Derive simple metrics ----
    cv_results = training_report.get("cv_results") or []
    cv_accs = [fold.get("accuracy", 0.0) for fold in cv_results if "accuracy" in fold]
    cv_f1s = [fold.get("f1", fold.get("f1_score", 0.0)) for fold in cv_results if "f1" in fold or "f1_score" in fold]

    avg_cv_acc = float(mean(cv_accs)) if cv_accs else 0.0
    avg_cv_f1 = float(mean(cv_f1s)) if cv_f1s else 0.0

    bt_metrics = backtest_report.get("metrics", {})
    net_pnl = float(bt_metrics.get("net_total_pnl", 0.0))
    sharpe = float(bt_metrics.get("sharpe_ratio", 0.0))
    net_dd = float(bt_metrics.get("net_max_drawdown", 0.0))

    # ---- Thresholds (absolute, can be tuned later) ----
    thresholds = {
        "min_cv_accuracy": 0.55,
        "min_cv_f1": 0.55,
        "min_sharpe": 0.8,
        "min_net_pnl": 0.0,
        "max_drawdown_abs": None,  # optional cap
    }

    checks = {
        "cv_accuracy_ok": avg_cv_acc >= thresholds["min_cv_accuracy"],
        "cv_f1_ok": avg_cv_f1 >= thresholds["min_cv_f1"],
        "sharpe_ok": sharpe >= thresholds["min_sharpe"],
        "net_pnl_ok": net_pnl >= thresholds["min_net_pnl"],
    }
    if thresholds["max_drawdown_abs"] is not None:
        checks["drawdown_ok"] = abs(net_dd) <= thresholds["max_drawdown_abs"]

    overall_pass = all(checks.values())

    eval_payload = {
        "exchange": ex,
        "tag": args.tag,
        "overall_pass": overall_pass,
        "checks": checks,
        "thresholds": thresholds,
        "cv": {
            "folds": len(cv_results),
            "avg_accuracy": avg_cv_acc,
            "avg_f1": avg_cv_f1,
        },
        "backtest": {
            "net_total_pnl": net_pnl,
            "sharpe_ratio": sharpe,
            "net_max_drawdown": net_dd,
        },
        "auto_ml_summary_present": bool(auto_ml_summary),
    }

    out_path = reports_dir / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    logging.info("Evaluation written to %s (overall_pass=%s)", out_path, overall_pass)


if __name__ == "__main__":
    main()


