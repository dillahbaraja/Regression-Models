import argparse

import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_pair_data
from feature_engineering import calculate_technical_features
from optimize_hyperparameter import optimize_hyperparameters
from target_builder import build_target


BASE_TUNING_MODELS = ["xgboost", "lightgbm", "catboost", "sklearn_hgb"]


def prepare_training_frame(symbol):
    timeframe = config.resolve_pair_timeframe(symbol)
    df = load_pair_data(symbol, timeframe=timeframe)
    df_features = calculate_technical_features(df)
    df_target = build_target(
        df_features,
        horizon=config.PREDICTION_HORIZON,
        target_type=config.TARGET_TYPE,
    )
    df_target = df_target.dropna().copy()

    if config.TRAIN_TEST_SPLIT_MODE == "date":
        train_df = df_target[df_target.index < config.TRAIN_END_DATE]
    else:
        split_idx = int(len(df_target) * (1 - config.TEST_SIZE))
        train_df = df_target.iloc[:split_idx]

    features = [col for col in config.FEATURE_COLUMNS if col in train_df.columns]
    if not features:
        raise ValueError(f"No usable features found for {symbol}.")
    if train_df.empty:
        raise ValueError(f"No training data available for {symbol}.")

    X_train = train_df[features]
    y_train = train_df["target"]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    return timeframe, X_train_scaled, y_train


def run_tuning_only(symbols=None, model_types=None):
    config.ensure_directories()
    symbols = symbols or config.REPORT_SYMBOLS
    model_types = model_types or BASE_TUNING_MODELS

    for symbol in symbols:
        print(f"Starting tuning-only pipeline for {symbol}...")
        timeframe, X_train_scaled, y_train = prepare_training_frame(symbol)
        for model_type in model_types:
            print(f"Running tuning-only for {symbol} / {model_type}...")
            best_params = optimize_hyperparameters(
                X_train_scaled,
                y_train,
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe,
            )
            print(f"Best params for {symbol} / {model_type}: {best_params}")
        print(f"Tuning-only pipeline completed for {symbol}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning only and write tuning logs without training final models."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(config.REPORT_SYMBOLS),
        help="Symbols to process. Default: report symbols only.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=BASE_TUNING_MODELS,
        help="Model types to tune. Default: boosting regression families.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tuning_only(symbols=args.symbols, model_types=args.models)
