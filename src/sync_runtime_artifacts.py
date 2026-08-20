from pathlib import Path
import shutil

import joblib
import pandas as pd

import config


def _copy_if_exists(source: Path, destinations):
    if not source.exists():
        return
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def export_scaler_csv_from_pickle(symbol, timeframe):
    scaler_path = config.scaler_path(symbol, timeframe)
    feature_path = config.feature_schema_path(symbol, timeframe)
    if not scaler_path.exists() or not feature_path.exists():
        return False

    scaler = joblib.load(scaler_path)
    features = pd.read_json(feature_path, typ="series").tolist()
    scaler_df = pd.DataFrame(
        {
            "feature": features,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    )
    filename = config.scaler_params_project_path(symbol, timeframe).name
    for path in (
        config.scaler_params_project_path(symbol, timeframe),
        config.scaler_params_terminal_path(symbol, timeframe),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_df.to_csv(path, index=False)
    print(f"Scaler params synced for {symbol} ({timeframe})")
    return True


def sync_symbol(symbol):
    timeframe = config.resolve_pair_timeframe(symbol)
    export_scaler_csv_from_pickle(symbol, timeframe)

    for model_type in config.MODEL_TYPES:
        onnx_source = config.onnx_model_path(symbol, timeframe, model_type)
        _copy_if_exists(
            onnx_source,
            [
                config.terminal_onnx_model_path(symbol, timeframe, model_type),
            ],
        )

        signal_source = config.signal_path(symbol, timeframe, model_type)
        _copy_if_exists(
            signal_source,
            [
                config.terminal_signal_path(symbol, timeframe, model_type),
            ],
        )


def main():
    config.ensure_directories()
    for symbol in config.training_symbols():
        sync_symbol(symbol)


if __name__ == "__main__":
    main()
