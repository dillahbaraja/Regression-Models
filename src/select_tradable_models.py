import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config


DEFAULT_INPUT_DIRS = [
    config.COMMON_FILES_DIR / "Regression_Test_Results",
    config.COMMON_REGRESSION_ASSETS_DIR,
    config.REPORTS_DIR / "backtest",
]
DEFAULT_OUTPUT_DIRS = [
    config.COMMON_FILES_DIR / "Regression_Model_Selection",
    config.REPORTS_DIR / "selection",
]
MODEL_ALIASES = {
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "catboost": "catboost",
    "sklearn_hgb": "sklearn_hgb",
    "hgb": "sklearn_hgb",
    "voting": "voting",
    "ensemble": "voting",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select tradable RegressionEA models from external test or backtest CSV results."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        help="Folder containing external test CSV files. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        help="Folder for selected_models.csv and rejected_models.csv. Defaults to Common Files.",
    )
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--min-expectancy", type=float, default=0.0)
    parser.add_argument("--min-win-rate", type=float, default=45.0)
    parser.add_argument("--max-drawdown", type=float, default=0.20)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folders recursively.",
    )
    return parser.parse_args()


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_column(df, candidates):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def parse_money_series(series):
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def infer_symbol_model(path, df):
    symbol_col = find_column(df, ["SYMBOL", "symbol"])
    symbol = None
    if symbol_col and not df[symbol_col].dropna().empty:
        symbol = str(df[symbol_col].dropna().iloc[0]).strip().upper()

    stem = path.stem
    if symbol is None:
        match = re.search(r"([A-Z]{6}|XAUUSD|XAGUSD)", stem.upper())
        if match:
            symbol = match.group(1)

    model = None
    for token, alias in MODEL_ALIASES.items():
        if re.search(rf"(^|[_\-\s]){re.escape(token)}($|[_\-\s])", stem.lower()):
            model = alias
            break

    if model is None:
        model_col = find_column(df, ["MODEL", "MODEL_NAME", "model", "model_type"])
        if model_col and not df[model_col].dropna().empty:
            raw_model = str(df[model_col].dropna().iloc[0]).strip().lower()
            model = MODEL_ALIASES.get(raw_model, raw_model)

    return symbol or "UNKNOWN", model or "UNKNOWN"


def returns_from_transaction_file(df):
    profit_col = find_column(df, ["PROFIT", "profit"])
    if not profit_col:
        return None

    entry_col = find_column(df, ["ENTRY", "entry"])
    working = df.copy()
    if entry_col:
        exits = working[entry_col].astype(str).str.upper().isin(["OUT", "OUT_BY", "INOUT"])
        if exits.any():
            working = working.loc[exits].copy()

    returns = parse_money_series(working[profit_col])
    for optional_col in ("SWAP", "COMMISSION"):
        col = find_column(working, [optional_col, optional_col.lower()])
        if col:
            returns = returns + parse_money_series(working[col])
    return returns


def returns_from_signal_file(df):
    profit_col = find_column(df, ["NET_PROFIT", "PROFIT", "profit", "net_profit", "strategy_return"])
    if profit_col:
        return parse_money_series(df[profit_col])

    prediction_col = find_column(df, ["prediction", "PREDICTION"])
    target_col = find_column(df, ["target", "TARGET"])
    if not prediction_col or not target_col:
        return None

    buy_threshold = float(config.THRESHOLD_BUY)
    sell_threshold = float(config.THRESHOLD_SELL)
    pred = pd.to_numeric(df[prediction_col], errors="coerce")
    target = pd.to_numeric(df[target_col], errors="coerce")
    signals = pd.Series(0, index=df.index, dtype=float)
    signals.loc[pred >= buy_threshold] = 1.0
    signals.loc[pred <= sell_threshold] = -1.0
    return (signals * target).fillna(0.0)


def max_drawdown_from_returns(returns):
    if returns.empty:
        return 0.0
    equity = returns.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return abs(float(drawdown.min()))


def classify_result(metrics, args):
    reasons = []
    if metrics["trade_count"] < args.min_trades:
        reasons.append(f"trade_count<{args.min_trades}")
    if metrics["profit_factor"] < args.min_profit_factor:
        reasons.append(f"profit_factor<{args.min_profit_factor}")
    if metrics["expectancy"] <= args.min_expectancy:
        reasons.append(f"expectancy<={args.min_expectancy}")
    if metrics["win_rate"] < args.min_win_rate:
        reasons.append(f"win_rate<{args.min_win_rate}")
    if metrics["max_drawdown"] > args.max_drawdown:
        reasons.append(f"max_drawdown>{args.max_drawdown}")
    return ("APPROVED" if not reasons else "REJECTED", ";".join(reasons))


def metrics_from_returns(path, df, returns, args):
    returns = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns[returns != 0]
    symbol, model = infer_symbol_model(path, df)

    trade_count = int(len(returns))
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(abs(returns[returns < 0].sum()))
    net_profit = float(returns.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
    win_rate = float((returns > 0).mean() * 100.0) if trade_count else 0.0
    expectancy = float(returns.mean()) if trade_count else 0.0
    max_drawdown = max_drawdown_from_returns(returns)

    metrics = {
        "STATUS": "",
        "REASON": "",
        "SYMBOL": symbol,
        "MODEL": model,
        "SOURCE_FILE": str(path),
        "trade_count": trade_count,
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
    }
    metrics["STATUS"], metrics["REASON"] = classify_result(metrics, args)
    return metrics


def discover_csv_files(input_dirs, recursive):
    files = []
    for folder in input_dirs:
        folder = Path(folder)
        if not folder.exists():
            continue
        pattern = "**/*.csv" if recursive else "*.csv"
        files.extend(sorted(folder.glob(pattern)))
    return list(dict.fromkeys(files))


def evaluate_file(path, args):
    try:
        df = normalize_columns(pd.read_csv(path))
    except Exception as exc:
        return {
            "STATUS": "ERROR",
            "REASON": f"read_failed:{exc}",
            "SYMBOL": "UNKNOWN",
            "MODEL": "UNKNOWN",
            "SOURCE_FILE": str(path),
        }

    returns = returns_from_transaction_file(df)
    if returns is None:
        returns = returns_from_signal_file(df)
    if returns is None:
        return {
            "STATUS": "SKIPPED",
            "REASON": "no_profit_or_prediction_target_columns",
            "SYMBOL": infer_symbol_model(path, df)[0],
            "MODEL": infer_symbol_model(path, df)[1],
            "SOURCE_FILE": str(path),
        }
    return metrics_from_returns(path, df, returns, args)


def write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "STATUS",
                "REASON",
                "SYMBOL",
                "MODEL",
                "SOURCE_FILE",
                "trade_count",
                "net_profit",
                "gross_profit",
                "gross_loss",
                "profit_factor",
                "win_rate",
                "expectancy",
                "max_drawdown",
            ]
        )

    selected = df[df["STATUS"] == "APPROVED"].copy()
    rejected = df[df["STATUS"] != "APPROVED"].copy()
    selected.to_csv(output_dir / "selected_models.csv", index=False)
    rejected.to_csv(output_dir / "rejected_models.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "approved_count": int(len(selected)),
        "rejected_count": int(len(rejected)),
        "generated_files": [
            str(output_dir / "selected_models.csv"),
            str(output_dir / "rejected_models.csv"),
        ],
    }
    with open(output_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args = parse_args()
    input_dirs = [Path(p) for p in args.input_dir] if args.input_dir else DEFAULT_INPUT_DIRS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIRS[0]

    csv_files = discover_csv_files(input_dirs, args.recursive)
    results = [evaluate_file(path, args) for path in csv_files]
    summary = write_outputs(results, output_dir)

    project_output_dir = DEFAULT_OUTPUT_DIRS[1]
    if project_output_dir.resolve() != output_dir.resolve():
        write_outputs(results, project_output_dir)

    print("Model selection completed.")
    print(f"Input CSV files scanned: {len(csv_files)}")
    print(f"Approved: {summary['approved_count']}")
    print(f"Rejected/skipped/error: {summary['rejected_count']}")
    print(f"Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()
