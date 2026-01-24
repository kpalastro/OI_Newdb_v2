#!/bin/bash
# Train/Test Split Example for Option Return Models
# This script demonstrates how to train on known data and test on untrained data

EXCHANGE="NSE"
TRAIN_START="2026-01-01"
TRAIN_END="2026-01-20"
TEST_START="2026-01-21"
TEST_END="2026-01-23"

echo "=========================================="
echo "Option Return Model: Train/Test Split"
echo "=========================================="
echo ""
echo "Training Period: $TRAIN_START to $TRAIN_END"
echo "Testing Period: $TEST_START to $TEST_END"
echo ""

# Step 1: Train on training period
echo "Step 1: Training models on training data..."
python3 train_option_return_models.py \
    --exchange $EXCHANGE \
    --start-date $TRAIN_START \
    --end-date $TRAIN_END

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "✓ Training complete"
echo ""

# Step 2: Check training summary
echo "Step 2: Training Summary:"
cat models/option_returns/$EXCHANGE/training_summary.json | python3 -m json.tool | grep -E "(training_start_date|training_end_date|n_samples|avg_r2|avg_direction_accuracy)" | head -10

echo ""
echo "Step 3: Backtesting on untrained data..."
python3 backtesting/option_return_backtest.py \
    --exchange $EXCHANGE \
    --start $TEST_START \
    --end $TEST_END \
    --min-confidence 0.4 \
    --min-return 0.5 \
    --output test_results_${TEST_START}_${TEST_END}.json

if [ $? -ne 0 ]; then
    echo "ERROR: Backtesting failed"
    exit 1
fi

echo ""
echo "✓ Backtesting complete"
echo ""
echo "Results saved to: test_results_${TEST_START}_${TEST_END}.json"
echo ""
echo "=========================================="
echo "Train/Test Split Complete"
echo "=========================================="
