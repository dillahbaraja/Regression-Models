import json
import joblib

import config
from train_model import export_onnx_model


def export_existing_onnx():
    config.ensure_directories()

    for symbol in config.training_symbols():
        timeframe = config.resolve_pair_timeframe(symbol)
        feature_path = config.feature_schema_path(symbol, timeframe)
        scaler_path = config.scaler_path(symbol, timeframe)

        if not feature_path.exists() or not scaler_path.exists():
            print(f"Skipping {symbol}: missing feature schema or scaler.")
            continue

        with open(feature_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        scaler = joblib.load(scaler_path)

        for model_type in config.MODEL_TYPES:
            model_path = config.model_path(symbol, timeframe, model_type)
            if not model_path.exists():
                print(f"Skipping {symbol} / {model_type}: model file not found.")
                continue

            model = joblib.load(model_path)
            print(f"Exporting ONNX for {symbol} / {model_type}...")
            export_onnx_model(symbol, timeframe, scaler, model, feature_names, model_type)


if __name__ == "__main__":
    export_existing_onnx()
