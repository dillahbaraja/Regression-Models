import json
import os

import numpy as np
import pandas as pd

import config


def _write_hybrid_signal_outputs(signal_df, timeframe, model_type):
    filename = config.hybrid_signal_path(timeframe, model_type).name
    target_paths = [
        config.hybrid_signal_path(timeframe, model_type),
        config.terminal_hybrid_signal_path(timeframe, model_type),
    ]
    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        signal_df.to_csv(path, index=False)
    return target_paths[0]


def _load_metrics(symbol, model_type):
    timeframe = config.resolve_pair_timeframe(symbol)
    metrics_file = config.metrics_path(symbol, timeframe, model_type)
    if not os.path.exists(metrics_file):
        return None
    metrics_df = pd.read_csv(metrics_file)
    if metrics_df.empty:
        return None
    return metrics_df.iloc[0].to_dict()


def _resolve_weights(symbols, model_type):
    manual_weights = {
        symbol: float(weight)
        for symbol, weight in config.PAIR_WEIGHTS.items()
        if symbol in symbols
    }
    if manual_weights and config.HYBRID_WEIGHT_MODE == "manual":
        total = sum(manual_weights.values())
        if total > 0:
            return {symbol: manual_weights.get(symbol, 0.0) / total for symbol in symbols}

    if config.HYBRID_WEIGHT_MODE == "equal":
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    # Default: inverse RMSE, fallback to equal when metrics are missing.
    raw_weights = {}
    for symbol in symbols:
        metrics = _load_metrics(symbol, model_type)
        rmse_value = None if not metrics else metrics.get("RMSE")
        if rmse_value is not None and pd.notna(rmse_value) and float(rmse_value) > 0:
            rmse = float(rmse_value)
            raw_weights[symbol] = 1.0 / max(rmse, 1e-12)
        elif symbol in manual_weights:
            raw_weights[symbol] = manual_weights[symbol]
        else:
            raw_weights[symbol] = 1.0

    total = sum(raw_weights.values())
    if total <= 0:
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}

    return {symbol: raw_weights[symbol] / total for symbol in symbols}


def _load_pair_predictions(symbol, model_type):
    timeframe = config.resolve_pair_timeframe(symbol)
    pred_file = config.predictions_path(symbol, timeframe, model_type)
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Missing predictions file for {symbol}: {pred_file}")
    df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
    df = df.rename(columns={
        "target": f"{symbol}_target",
        "prediction": f"{symbol}_prediction",
    })
    return df[[f"{symbol}_target", f"{symbol}_prediction"]]


def build_hybrid_predictions(timeframe=None):
    report_symbols = list(config.report_symbols())
    timeframe = timeframe or config.resolve_pair_timeframe(report_symbols[0])
    config.ensure_directories()

    symbols = report_symbols
    
    all_metrics = []
    
    for model_type in config.MODEL_TYPES:
        weights = _resolve_weights(symbols, model_type)
    
        merged = None
        for symbol in symbols:
            try:
                pair_df = _load_pair_predictions(symbol, model_type)
                merged = pair_df if merged is None else merged.join(pair_df, how="inner")
            except FileNotFoundError:
                continue
                
        if merged is None:
            continue
    
        target_cols = [f"{symbol}_target" for symbol in symbols if f"{symbol}_target" in merged.columns]
        pred_cols = [f"{symbol}_prediction" for symbol in symbols if f"{symbol}_prediction" in merged.columns]
        actual_symbols = [s for s in symbols if f"{s}_target" in merged.columns]
        
        if not actual_symbols:
            continue
    
        weight_array = np.array([weights[symbol] for symbol in actual_symbols], dtype=float)
        # Normalize weights for the actual symbols present
        if weight_array.sum() > 0:
            weight_array = weight_array / weight_array.sum()
            
        target_matrix = merged[target_cols].to_numpy(dtype=float)
        pred_matrix = merged[pred_cols].to_numpy(dtype=float)
    
        hybrid_df = pd.DataFrame(index=merged.index)
        hybrid_df["target"] = target_matrix.dot(weight_array)
        hybrid_df["prediction"] = pred_matrix.dot(weight_array)
        hybrid_df["source_count"] = len(actual_symbols)
        hybrid_df["timeframe"] = timeframe
        hybrid_df["symbol"] = config.HYBRID_SYMBOL
    
        hybrid_df.to_csv(config.hybrid_predictions_path(timeframe, model_type))
    
        rmse = float(np.sqrt(np.mean((hybrid_df["target"] - hybrid_df["prediction"]) ** 2)))
        mae = float(np.mean(np.abs(hybrid_df["target"] - hybrid_df["prediction"])))
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(hybrid_df["target"], hybrid_df["prediction"]))
        except Exception:
            r2 = np.nan
    
        dir_acc = np.nan
        if config.TARGET_TYPE == "return":
            dir_acc = float(np.mean((hybrid_df["target"] > 0) == (hybrid_df["prediction"] > 0)) * 100)
    
        metrics_df = pd.DataFrame({
            "symbol": [config.HYBRID_SYMBOL],
            "timeframe": [timeframe],
            "model_type": [model_type],
            "RMSE": [rmse],
            "MAE": [mae],
            "R2": [r2],
            "Directional_Accuracy": [dir_acc],
        })
        metrics_df.to_csv(config.hybrid_metrics_path(timeframe, model_type), index=False)
        all_metrics.append(metrics_df)
    
        signal_rows = []
        for time, row in hybrid_df.iterrows():
            pred = row["prediction"]
            if config.TARGET_TYPE == "return":
                if pred >= config.THRESHOLD_BUY:
                    signal = "BUY"
                elif pred <= config.THRESHOLD_SELL:
                    signal = "SELL"
                else:
                    signal = "NONE"
            else:
                signal = "NONE"
    
            signal_rows.append({
                "DATETIME": time.strftime("%Y.%m.%d %H:%M:%S"),
                "SYMBOL": config.HYBRID_SYMBOL,
                "TIMEFRAME": timeframe,
                "SIGNAL": signal,
                "PREDICTION": pred,
                "CONFIDENCE": abs(pred),
                "SL": 0.0,
                "TP": 0.0,
                "WEIGHTS": json.dumps({s: weights[s] for s in actual_symbols}, sort_keys=True),
            })
    
        signal_df = pd.DataFrame(signal_rows)
        _write_hybrid_signal_outputs(signal_df, timeframe, model_type)
    
        weights_path = config.hybrid_weights_path(timeframe) # weights can remain un-versioned or versioned, let's keep it un-versioned for now to save space, or just print it.
    
        print(f"[{model_type}] Hybrid predictions saved.")

    return all_metrics


if __name__ == "__main__":
    build_hybrid_predictions()
