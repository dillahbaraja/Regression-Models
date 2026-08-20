import pandas as pd
import numpy as np
import config
import os


def load_thresholds(symbol, timeframe, model_type):
    threshold_file = config.thresholds_path(symbol, timeframe, model_type)
    if not os.path.exists(threshold_file):
        return float(config.THRESHOLD_BUY), float(config.THRESHOLD_SELL)

    df = pd.read_csv(threshold_file)
    if df.empty:
        return float(config.THRESHOLD_BUY), float(config.THRESHOLD_SELL)

    row = df.iloc[0]
    return float(row["BUY_THRESHOLD"]), float(row["SELL_THRESHOLD"])


def backtest_predictions_file(pred_file, output_file, buy_threshold=None, sell_threshold=None):
    print(f"Running Python Backtest for {pred_file}...")
    
    if not os.path.exists(pred_file):
        print(f"Predictions file not found at {pred_file}")
        return
        
    df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
    
    # Assuming target is actual forward return
    # If signal is BUY, we capture target. If SELL, we capture -target
    
    df['signal'] = 0
    active_buy_threshold = float(config.THRESHOLD_BUY if buy_threshold is None else buy_threshold)
    active_sell_threshold = float(config.THRESHOLD_SELL if sell_threshold is None else sell_threshold)
    df.loc[df['prediction'] >= active_buy_threshold, 'signal'] = 1
    df.loc[df['prediction'] <= active_sell_threshold, 'signal'] = -1
    
    df['strategy_return'] = df['signal'] * df['target']
    
    df['equity_curve'] = (1 + df['strategy_return']).cumprod()
    df['buy_and_hold'] = (1 + df['target']).cumprod()
    
    total_return = df['equity_curve'].iloc[-1] - 1
    win_rate = np.mean(df[df['signal'] != 0]['strategy_return'] > 0) * 100
    
    print(f"Total Strategy Return: {total_return * 100:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Trades: {sum(df['signal'] != 0)}")
    
    df[['equity_curve', 'buy_and_hold']].to_csv(output_file)
    return df


def simple_python_backtest_for_symbol(symbol):
    timeframe = config.resolve_pair_timeframe(symbol)
    results = []
    for model_type in config.MODEL_TYPES:
        buy_threshold, sell_threshold = load_thresholds(symbol, timeframe, model_type)
        res = backtest_predictions_file(
            config.predictions_path(symbol, timeframe, model_type),
            config.backtest_path(symbol, timeframe, model_type),
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        if res is not None:
            results.append(res)
    return results


def simple_python_backtest_hybrid(timeframe=None):
    report_symbols = config.report_symbols()
    timeframe = timeframe or config.resolve_pair_timeframe(report_symbols[0])
    results = []
    for model_type in config.MODEL_TYPES:
        res = backtest_predictions_file(
            config.hybrid_predictions_path(timeframe, model_type),
            config.backtest_path(config.HYBRID_SYMBOL, timeframe, model_type),
        )
        if res is not None:
            results.append(res)
    return results


def simple_python_backtest_all():
    config.ensure_directories()
    results = []
    for symbol in config.report_symbols():
        results.append(simple_python_backtest_for_symbol(symbol))
    report_symbols = config.report_symbols()
    hybrid_timeframe = config.resolve_pair_timeframe(report_symbols[0])
    # Hybrid predictions should have been built for all models, so we can just call it
    res = simple_python_backtest_hybrid(hybrid_timeframe)
    results.extend(res)
    return results

if __name__ == "__main__":
    simple_python_backtest_all()
