# Reverse Engineering Trade Logs to Improve Model Efficiency

## Overview

This guide explains how to analyze trade logs to identify patterns, extract insights, and improve model performance through reverse engineering successful and failed trades.

## Available Data Sources

### 1. Trade Logs (`trade_logs/trades_YYYY-MM-DD.csv`)

**Columns Available:**
- `entry_timestamp`, `exit_timestamp` - Timing information
- `exchange` - NSE, BSE, etc.
- `position_id` - Unique trade identifier
- `symbol` - Option symbol
- `type` - CE (Call) or PE (Put)
- `side` - BUY or SELL
- `quantity` - Number of contracts
- `entry_price`, `exit_price` - Entry and exit prices
- `pnl` - Profit/Loss (realized)
- `entry_reason` - Why trade was entered (e.g., "Auto-lightgbm-LimitChase")
- `exit_reason` - Why trade was exited (e.g., "Target Hit (+25)", "Stop Loss")
- `status` - OPEN or CLOSED
- `confidence` - Model confidence at entry (0-1)
- `kelly_fraction` - Position sizing fraction
- `constraint_violation` - Whether risk constraints were violated
- `signal_id` - Links to recommendation logs

### 2. Recommendation Logs (`logs/recommendations/YYYY-MM-DD.jsonl`)

**Contains:**
- `timestamp` - When signal was generated
- `exchange` - Exchange name
- `signal` - BUY, SELL, HOLD
- `confidence` - Model confidence
- `regime` - Market regime (e.g., "HIGH_VOL", "LOW_VOL_COMPRESSION")
- `kelly_fraction` - Recommended position size
- `recommended_lots` - Number of lots
- `model_version` - Model version used
- `signal_id` - Links to trade logs

### 3. Database Tables

- `ml_features` - Historical feature values at prediction time
- `paper_trading_orders` - All paper trading executions
- `ml_performance` - Model performance metrics

---

## Step-by-Step Reverse Engineering Process

### Step 1: Load and Analyze Trade Logs

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def load_trade_logs(days=30):
    """Load trade logs for the last N days."""
    log_dir = Path("trade_logs")
    dfs = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        log_file = log_dir / f"trades_{date.strftime('%Y-%m-%d')}.csv"
        if log_file.exists():
            df = pd.read_csv(log_file)
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert timestamps
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
    
    # Calculate holding period
    df['holding_period_minutes'] = (
        (df['exit_timestamp'] - df['entry_timestamp']).dt.total_seconds() / 60
    )
    
    # Calculate return percentage
    df['return_pct'] = (df['pnl'] / (df['entry_price'] * df['quantity'])) * 100
    
    # Classify trades
    df['is_winner'] = df['pnl'] > 0
    df['is_large_winner'] = df['pnl'] > df['pnl'].quantile(0.75)
    df['is_large_loser'] = df['pnl'] < df['pnl'].quantile(0.25)
    
    return df

# Load data
trades = load_trade_logs(days=30)
```

### Step 2: Identify Winning Patterns

```python
def analyze_winning_patterns(trades):
    """Identify patterns in winning trades."""
    
    # Filter closed trades only
    closed = trades[trades['status'] == 'CLOSED'].copy()
    
    if len(closed) == 0:
        print("No closed trades found")
        return
    
    winners = closed[closed['is_winner'] == True]
    losers = closed[closed['is_winner'] == False]
    
    print(f"Total Trades: {len(closed)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(closed)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(closed)*100:.1f}%)")
    print(f"\nAverage PnL: {closed['pnl'].mean():.2f}")
    print(f"Average Winner: {winners['pnl'].mean():.2f}")
    print(f"Average Loser: {losers['pnl'].mean():.2f}")
    
    # Analyze by confidence level
    print("\n=== Win Rate by Confidence Level ===")
    closed['conf_bucket'] = pd.cut(closed['confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0])
    conf_analysis = closed.groupby('conf_bucket').agg({
        'is_winner': ['mean', 'count'],
        'pnl': 'mean'
    })
    print(conf_analysis)
    
    # Analyze by regime
    print("\n=== Win Rate by Regime ===")
    # Need to join with recommendation logs for regime
    # This is a placeholder - you'd need to load recommendation logs
    # regime_analysis = closed.groupby('regime').agg({
    #     'is_winner': ['mean', 'count'],
    #     'pnl': 'mean'
    # })
    
    # Analyze by entry reason
    print("\n=== Win Rate by Entry Reason ===")
    entry_analysis = closed.groupby('entry_reason').agg({
        'is_winner': ['mean', 'count'],
        'pnl': 'mean',
        'confidence': 'mean'
    }).sort_values(('is_winner', 'mean'), ascending=False)
    print(entry_analysis.head(10))
    
    # Analyze by exit reason
    print("\n=== Exit Reason Analysis ===")
    exit_analysis = closed.groupby('exit_reason').agg({
        'is_winner': ['mean', 'count'],
        'pnl': 'mean',
        'holding_period_minutes': 'mean'
    }).sort_values(('is_winner', 'mean'), ascending=False)
    print(exit_analysis)
    
    # Analyze by time of day
    print("\n=== Win Rate by Entry Hour ===")
    closed['entry_hour'] = closed['entry_timestamp'].dt.hour
    hour_analysis = closed.groupby('entry_hour').agg({
        'is_winner': ['mean', 'count'],
        'pnl': 'mean'
    })
    print(hour_analysis)
    
    return {
        'winners': winners,
        'losers': losers,
        'closed': closed
    }

results = analyze_winning_patterns(trades)
```

### Step 3: Join with Feature Data

```python
import database_new as db

def enrich_trades_with_features(trades):
    """Join trade logs with historical feature data."""
    
    enriched = []
    
    for _, trade in trades.iterrows():
        if pd.isna(trade['signal_id']):
            continue
            
        # Extract timestamp from signal_id or use entry_timestamp
        timestamp = trade['entry_timestamp']
        
        # Query features at entry time
        try:
            features = db.get_features_at_timestamp(
                exchange=trade['exchange'],
                timestamp=timestamp
            )
            
            if features:
                trade_dict = trade.to_dict()
                trade_dict.update(features)
                enriched.append(trade_dict)
        except Exception as e:
            print(f"Error loading features for {trade['position_id']}: {e}")
            continue
    
    return pd.DataFrame(enriched)

# Enrich trades with features
enriched_trades = enrich_trades_with_features(trades)
```

### Step 4: Feature Importance Analysis

```python
def analyze_feature_importance(enriched_trades):
    """Identify which features correlate with winning trades."""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    feature_cols = [col for col in enriched_trades.columns 
                   if col not in ['position_id', 'entry_timestamp', 'exit_timestamp',
                                 'symbol', 'entry_reason', 'exit_reason', 'status',
                                 'is_winner', 'pnl', 'return_pct', 'signal_id']]
    
    X = enriched_trades[feature_cols].select_dtypes(include=[np.number])
    y = enriched_trades['is_winner'].astype(int)
    
    # Remove NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        print("No valid data for feature importance analysis")
        return
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    
    # Train simple classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top Features Correlating with Winning Trades ===")
    print(importance_df.head(20))
    
    # Analyze feature values for winners vs losers
    print("\n=== Feature Value Comparison: Winners vs Losers ===")
    winners = enriched_trades[enriched_trades['is_winner'] == True]
    losers = enriched_trades[enriched_trades['is_winner'] == False]
    
    top_features = importance_df.head(10)['feature'].tolist()
    
    for feat in top_features:
        if feat in enriched_trades.columns:
            winner_mean = winners[feat].mean()
            loser_mean = losers[feat].mean()
            diff = winner_mean - loser_mean
            print(f"{feat:30s} | Winners: {winner_mean:8.4f} | Losers: {loser_mean:8.4f} | Diff: {diff:8.4f}")
    
    return importance_df

feature_importance = analyze_feature_importance(enriched_trades)
```

### Step 5: Identify Optimal Confidence Thresholds

```python
def find_optimal_confidence_threshold(trades):
    """Find the confidence threshold that maximizes win rate."""
    
    closed = trades[trades['status'] == 'CLOSED'].copy()
    
    if len(closed) == 0:
        return None
    
    thresholds = np.arange(0.4, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        filtered = closed[closed['confidence'] >= threshold]
        
        if len(filtered) == 0:
            continue
        
        win_rate = filtered['is_winner'].mean()
        total_trades = len(filtered)
        avg_pnl = filtered['pnl'].mean()
        total_pnl = filtered['pnl'].sum()
        
        results.append({
            'threshold': threshold,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'sharpe_like': avg_pnl / (filtered['pnl'].std() + 1e-6) if len(filtered) > 1 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n=== Optimal Confidence Threshold Analysis ===")
    print(results_df.to_string(index=False))
    
    # Find threshold with best win rate while maintaining reasonable trade count
    results_df['score'] = (
        results_df['win_rate'] * 0.4 + 
        (results_df['total_trades'] / results_df['total_trades'].max()) * 0.3 +
        (results_df['avg_pnl'] / (abs(results_df['avg_pnl']).max() + 1e-6)) * 0.3
    )
    
    optimal = results_df.loc[results_df['score'].idxmax()]
    print(f"\nRecommended Confidence Threshold: {optimal['threshold']:.2f}")
    print(f"  Win Rate: {optimal['win_rate']:.2%}")
    print(f"  Total Trades: {optimal['total_trades']:.0f}")
    print(f"  Avg PnL: {optimal['avg_pnl']:.2f}")
    
    return optimal

optimal_threshold = find_optimal_confidence_threshold(trades)
```

### Step 6: Analyze Regime-Specific Performance

```python
def analyze_regime_performance(trades, recommendation_logs):
    """Analyze performance by market regime."""
    
    # Load recommendation logs
    import json
    from pathlib import Path
    
    rec_dir = Path("logs/recommendations")
    rec_data = []
    
    for log_file in sorted(rec_dir.glob("*.jsonl")):
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    rec_data.append(json.loads(line))
                except:
                    continue
    
    rec_df = pd.DataFrame(rec_data)
    rec_df['timestamp'] = pd.to_datetime(rec_df['timestamp'])
    
    # Join with trades
    trades_with_regime = trades.merge(
        rec_df[['signal_id', 'regime', 'confidence']],
        on='signal_id',
        how='left',
        suffixes=('', '_rec')
    )
    
    closed = trades_with_regime[trades_with_regime['status'] == 'CLOSED']
    
    if 'regime' not in closed.columns or closed['regime'].isna().all():
        print("No regime data available")
        return
    
    print("\n=== Performance by Market Regime ===")
    regime_analysis = closed.groupby('regime').agg({
        'is_winner': ['mean', 'count'],
        'pnl': ['mean', 'sum'],
        'confidence': 'mean',
        'holding_period_minutes': 'mean'
    })
    
    print(regime_analysis)
    
    # Identify best and worst regimes
    best_regime = regime_analysis[('is_winner', 'mean')].idxmax()
    worst_regime = regime_analysis[('is_winner', 'mean')].idxmin()
    
    print(f"\nBest Performing Regime: {best_regime}")
    print(f"  Win Rate: {regime_analysis.loc[best_regime, ('is_winner', 'mean')]:.2%}")
    print(f"  Avg PnL: {regime_analysis.loc[best_regime, ('pnl', 'mean')]:.2f}")
    
    print(f"\nWorst Performing Regime: {worst_regime}")
    print(f"  Win Rate: {regime_analysis.loc[worst_regime, ('is_winner', 'mean')]:.2%}")
    print(f"  Avg PnL: {regime_analysis.loc[worst_regime, ('pnl', 'mean')]:.2f}")
    
    return regime_analysis

regime_performance = analyze_regime_performance(trades, None)
```

### Step 7: Create New Features from Trade Patterns

```python
def create_trade_pattern_features(trades):
    """Create new features based on trade log patterns."""
    
    # Feature 1: Time-based features
    trades['entry_hour'] = trades['entry_timestamp'].dt.hour
    trades['entry_day_of_week'] = trades['entry_timestamp'].dt.dayofweek
    trades['is_morning'] = trades['entry_hour'].between(9, 11)
    trades['is_afternoon'] = trades['entry_hour'].between(14, 15)
    
    # Feature 2: Confidence-distance features
    trades['confidence_distance_from_optimal'] = abs(
        trades['confidence'] - optimal_threshold['threshold']
    )
    
    # Feature 3: Historical performance features (rolling)
    trades = trades.sort_values('entry_timestamp')
    trades['rolling_win_rate_10'] = (
        trades['is_winner'].rolling(window=10, min_periods=1).mean()
    )
    trades['rolling_avg_pnl_10'] = (
        trades['pnl'].rolling(window=10, min_periods=1).mean()
    )
    
    # Feature 4: Entry reason encoding
    entry_reason_counts = trades['entry_reason'].value_counts()
    trades['entry_reason_frequency'] = trades['entry_reason'].map(entry_reason_counts)
    
    return trades

trades_with_patterns = create_trade_pattern_features(trades)
```

### Step 8: Generate Improvement Recommendations

```python
def generate_improvement_recommendations(analysis_results):
    """Generate actionable recommendations based on analysis."""
    
    recommendations = []
    
    # Recommendation 1: Confidence threshold
    if optimal_threshold:
        current_threshold = 0.55  # Default from config
        if optimal_threshold['threshold'] > current_threshold + 0.05:
            recommendations.append({
                'type': 'confidence_threshold',
                'priority': 'HIGH',
                'current': current_threshold,
                'recommended': optimal_threshold['threshold'],
                'expected_improvement': f"Win rate: {optimal_threshold['win_rate']:.1%}",
                'action': f"Update config.min_confidence_for_trade to {optimal_threshold['threshold']:.2f}"
            })
    
    # Recommendation 2: Feature engineering
    if feature_importance is not None:
        top_features = feature_importance.head(5)['feature'].tolist()
        recommendations.append({
            'type': 'feature_engineering',
            'priority': 'MEDIUM',
            'recommendation': 'Focus on these top features in model training',
            'features': top_features,
            'action': 'Review feature_engineering.py and ensure these features are well-calculated'
        })
    
    # Recommendation 3: Regime-specific models
    if regime_performance is not None:
        worst_regime = regime_performance[('is_winner', 'mean')].idxmin()
        recommendations.append({
            'type': 'regime_specific_training',
            'priority': 'MEDIUM',
            'regime': worst_regime,
            'current_win_rate': regime_performance.loc[worst_regime, ('is_winner', 'mean')],
            'action': f'Consider training regime-specific models for {worst_regime}'
        })
    
    # Recommendation 4: Entry reason filtering
    if 'entry_analysis' in analysis_results:
        entry_analysis = analysis_results['entry_analysis']
        low_performing = entry_analysis[
            (entry_analysis[('is_winner', 'mean')] < 0.5) & 
            (entry_analysis[('is_winner', 'count')] > 5)
        ]
        
        if len(low_performing) > 0:
            recommendations.append({
                'type': 'entry_reason_filtering',
                'priority': 'LOW',
                'low_performing_reasons': low_performing.index.tolist(),
                'action': 'Consider filtering out or reducing confidence for these entry reasons'
            })
    
    return recommendations

recommendations = generate_improvement_recommendations({
    'optimal_threshold': optimal_threshold,
    'feature_importance': feature_importance,
    'regime_performance': regime_performance,
    'entry_analysis': entry_analysis if 'entry_analysis' in locals() else None
})

print("\n=== IMPROVEMENT RECOMMENDATIONS ===")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['type'].upper()} ({rec['priority']} Priority)")
    for key, value in rec.items():
        if key != 'type' and key != 'priority':
            print(f"   {key}: {value}")
```

---

## Implementation Workflow

### Weekly Analysis Routine

1. **Monday Morning (30 minutes)**
   - Load last week's trade logs
   - Run basic win rate analysis
   - Check for any obvious degradation

2. **Monthly Deep Dive (2 hours)**
   - Full feature importance analysis
   - Regime performance review
   - Confidence threshold optimization
   - Generate improvement recommendations

3. **Quarterly Model Update**
   - Incorporate findings into model training
   - Update feature engineering based on patterns
   - Adjust confidence thresholds
   - Retrain models with new insights

### Automated Monitoring

Create a script that runs daily:

```python
# scripts/daily_trade_analysis.py
def daily_trade_analysis():
    """Run daily analysis and alert on issues."""
    trades = load_trade_logs(days=7)
    
    if len(trades) == 0:
        return
    
    closed = trades[trades['status'] == 'CLOSED']
    
    if len(closed) == 0:
        return
    
    # Check win rate
    win_rate = closed['is_winner'].mean()
    
    if win_rate < 0.50:
        alert = f"WARNING: Win rate dropped to {win_rate:.1%} (threshold: 50%)"
        print(alert)
        # Send alert (email, Slack, etc.)
    
    # Check average PnL
    avg_pnl = closed['pnl'].mean()
    if avg_pnl < 0:
        alert = f"WARNING: Average PnL is negative: {avg_pnl:.2f}"
        print(alert)
```

---

## Key Insights to Extract

1. **Confidence Calibration**: Are high-confidence trades actually winning more?
2. **Regime Adaptation**: Which regimes work best/worst for the model?
3. **Timing Patterns**: Are there specific times when trades perform better?
4. **Feature Correlations**: Which features most strongly predict success?
5. **Entry/Exit Patterns**: What entry reasons and exit reasons correlate with wins?
6. **Position Sizing**: Is Kelly fraction optimal? Should it be adjusted?

---

## Next Steps

1. **Implement the analysis scripts** above
2. **Set up automated daily monitoring**
3. **Create a dashboard** to visualize findings
4. **Integrate findings into model training** pipeline
5. **A/B test improvements** before full deployment

---

## Example: Complete Analysis Script

Save this as `scripts/analyze_trade_logs.py`:

```python
#!/usr/bin/env python3
"""
Complete trade log analysis script.
Run: python scripts/analyze_trade_logs.py --days 30
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# [Include all the functions from above]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze trade logs')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--output', type=str, default='reports/trade_analysis.json', 
                       help='Output file for recommendations')
    
    args = parser.parse_args()
    
    print(f"Loading trade logs for last {args.days} days...")
    trades = load_trade_logs(days=args.days)
    
    if len(trades) == 0:
        print("No trade logs found")
        exit(1)
    
    print(f"Found {len(trades)} trades")
    
    # Run all analyses
    results = analyze_winning_patterns(trades)
    feature_importance = analyze_feature_importance(enriched_trades) if 'enriched_trades' in locals() else None
    optimal_threshold = find_optimal_confidence_threshold(trades)
    regime_performance = analyze_regime_performance(trades, None)
    recommendations = generate_improvement_recommendations({
        'optimal_threshold': optimal_threshold,
        'feature_importance': feature_importance,
        'regime_performance': regime_performance
    })
    
    # Save recommendations
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'days_analyzed': args.days,
            'total_trades': len(trades),
            'recommendations': recommendations
        }, f, indent=2)
    
    print(f"\nAnalysis complete. Recommendations saved to {output_path}")
```

---

This systematic approach will help you continuously improve your models by learning from actual trading performance!
