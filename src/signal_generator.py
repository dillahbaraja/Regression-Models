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


def _write_signal_outputs(signal_df, symbol, timeframe, model_type):
    filename = config.signal_path(symbol, timeframe, model_type).name
    target_paths = [
        config.signal_path(symbol, timeframe, model_type),
        config.terminal_signal_path(symbol, timeframe, model_type),
    ]
    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        signal_df.to_csv(path, index=False)
    return target_paths[0]


def generate_signals_for_symbol(symbol):
    timeframe = config.resolve_pair_timeframe(symbol)
    print(f"Generating Trading Signals for {symbol} ({timeframe})...")
    
    for model_type in config.MODEL_TYPES:
        pred_file = config.predictions_path(symbol, timeframe, model_type)
        if not os.path.exists(pred_file):
            continue
            
        df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
        buy_threshold, sell_threshold = load_thresholds(symbol, timeframe, model_type)
        signals = []
        
        for time, row in df.iterrows():
            pred = row['prediction']
            
            # Determine Signal based on Return Thresholds
            if config.TARGET_TYPE == "return":
                if pred >= buy_threshold:
                    signal = "BUY"
                elif pred <= sell_threshold:
                    signal = "SELL"
                else:
                    signal = "NONE"
            else:
                signal = "NONE"
                
            confidence = abs(pred)
            sl = 0.0
            tp = 0.0
            
            signals.append({
                "DATETIME": time.strftime("%Y.%m.%d %H:%M:%S"),
                "SYMBOL": symbol,
                "TIMEFRAME": timeframe,
                "SIGNAL": signal,
                "PREDICTION": pred,
                "CONFIDENCE": confidence,
                "SL": sl,
                "TP": tp
            })
        
        signal_df = pd.DataFrame(signals)
        
        out_path = _write_signal_outputs(signal_df, symbol, timeframe, model_type)
        
        print(f"[{model_type}] Signals saved to {out_path}")
        print(signal_df['SIGNAL'].value_counts())
        
    return None


def generate_all_signals():
    config.ensure_directories()
    frames = []
    for symbol in config.report_symbols():
        frame = generate_signals_for_symbol(symbol)
        if frame is not None:
            frames.append(frame)
    return frames

if __name__ == "__main__":
    generate_all_signals()
