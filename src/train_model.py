import pandas as pd
import numpy as np
import os
import json
import joblib
import shutil
import argparse
from contextlib import contextmanager
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import config
from sklearn.ensemble import VotingRegressor
from feature_engineering import calculate_technical_features
from news_features import build_news_feature_frame, load_news_data
from runtime_exporter import export_runtime_artifacts
from target_builder import build_target
from optimize_hyperparameter import optimize_hyperparameters
from data_loader import load_pair_data, load_data_from_csv

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
    from catboost import CatBoostError
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None
    CatBoostError = RuntimeError

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover - should exist in sklearn, but keep safe
    HistGradientBoostingRegressor = None

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:  # pragma: no cover - optional dependency
    convert_sklearn = None
    FloatTensorType = None

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxToolsFloatTensorType
except ImportError:  # pragma: no cover - optional dependency
    onnxmltools = None
    OnnxToolsFloatTensorType = None


@contextmanager
def patched_onnx_make_attribute():
    """
    skl2onnx may emit lists containing booleans for TreeEnsemble attributes.
    ONNX expects integer arrays there, so coerce bool values to int during export.
    """
    try:
        import onnx.helper as onnx_helper
    except ImportError:
        yield
        return

    original = onnx_helper.make_attribute

    def patched(key, value, *args, **kwargs):
        if isinstance(value, list):
            value = [int(item) if isinstance(item, bool) else item for item in value]
        return original(key, value, *args, **kwargs)

    onnx_helper.make_attribute = patched
    try:
        yield
    finally:
        onnx_helper.make_attribute = original


def write_onnx_to_targets(onnx_bytes, target_paths):
    target_paths = list(dict.fromkeys(target_paths))
    primary_path = target_paths[0]
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(primary_path, "wb") as f:
        f.write(onnx_bytes)

    for extra_path in target_paths[1:]:
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary_path, extra_path)

    return primary_path


def prepare_full_training_sources():
    """
    Mirror the current source datasets into the isolated full-data folders.
    This keeps the live-demo training inputs separate from the legacy split pipeline.
    """
    config.ensure_directories(variant="full")

    source_dataset_dir = config.DATASET_DIR
    target_ohlcv_dir = config.FULL_OHLCV_DATA_DIR
    target_news_dir = config.FULL_NEWS_DATA_DIR

    for src in sorted(source_dataset_dir.glob("*_Data.csv")):
        if src.name.upper().startswith("NEWS_"):
            shutil.copy2(src, target_news_dir / src.name)
        else:
            shutil.copy2(src, target_ohlcv_dir / src.name)

    news_source = config.NEWS_DATA_PATH
    if news_source.exists():
        shutil.copy2(news_source, target_news_dir / news_source.name)

    return target_ohlcv_dir, target_news_dir


def normalize_catboost_onnx_output(path):
    """
    CatBoost ONNX export may declare a 1D output in graph metadata while the
    runtime emits a [N, 1] tensor. MT5 is strict about shape consistency, so
    normalize the output metadata to two dimensions.
    """
    try:
        import onnx
    except ImportError:
        return

    model = onnx.load(str(path))
    if not model.graph.output:
        return

    output_tensor = model.graph.output[0].type.tensor_type
    dims = output_tensor.shape.dim
    if len(dims) == 1:
        extra_dim = dims.add()
        extra_dim.dim_value = 1
        onnx.save(model, str(path))


def export_xgboost_onnx(model, n_features):
    if onnxmltools is None or OnnxToolsFloatTensorType is None:
        raise ImportError("onnxmltools is not installed.")

    booster = model.get_booster()
    original_feature_names = booster.feature_names
    try:
        booster.feature_names = [f"f{i}" for i in range(n_features)]
        initial_types = [("input", OnnxToolsFloatTensorType([None, n_features]))]
        return onnxmltools.convert_xgboost(model, initial_types=initial_types).SerializeToString()
    finally:
        booster.feature_names = original_feature_names


def export_lightgbm_onnx(model, n_features):
    if onnxmltools is None or OnnxToolsFloatTensorType is None:
        raise ImportError("onnxmltools is not installed.")

    initial_types = [("input", OnnxToolsFloatTensorType([None, n_features]))]
    return onnxmltools.convert_lightgbm(model, initial_types=initial_types).SerializeToString()


def export_sklearn_regressor_onnx(model, n_features):
    if convert_sklearn is None or FloatTensorType is None:
        raise ImportError("skl2onnx is not installed.")

    initial_types = [("input", FloatTensorType([None, n_features]))]
    with patched_onnx_make_attribute():
        return convert_sklearn(
            model,
            initial_types=initial_types,
            target_opset=15,
        ).SerializeToString()


def build_model(model_type, best_params):
    if model_type == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install it or switch MODEL_TYPE.")
        return XGBRegressor(**best_params, random_state=42)
    if model_type == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed. Install it or switch MODEL_TYPE.")
        return LGBMRegressor(**best_params, random_state=42)
    if model_type == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed. Install it or switch MODEL_TYPE.")
        return CatBoostRegressor(
            **best_params,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
            allow_writing_files=False,
            thread_count=config.CATBOOST_THREAD_COUNT,
            used_ram_limit=config.CATBOOST_USED_RAM_LIMIT,
        )
    if model_type == "sklearn_hgb":
        if HistGradientBoostingRegressor is None:
            raise ImportError("HistGradientBoostingRegressor is unavailable in this sklearn build.")
        return HistGradientBoostingRegressor(**best_params, random_state=42)
    raise ValueError(f"Unknown model type: {model_type}")


def export_onnx_model(symbol, timeframe, scaler, model, feature_names, model_type, variant=None):
    n_features = len(feature_names)
    target_paths = [
        config.onnx_model_path(symbol, timeframe, model_type, variant=variant),
        config.terminal_onnx_model_path(symbol, timeframe, model_type, variant=variant),
    ]

    try:
        if model_type == "catboost":
            out_path = target_paths[0]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(out_path), format="onnx")
            normalize_catboost_onnx_output(out_path)
            for extra_path in target_paths[1:]:
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out_path, extra_path)
            print(f"ONNX model saved to {out_path}")
            print("Note: CatBoost ONNX contains the model only; use the saved scaler separately.")
            return out_path

        if model_type == "xgboost":
            onnx_bytes = export_xgboost_onnx(model, n_features)
            out_path = write_onnx_to_targets(onnx_bytes, target_paths)
            print(f"ONNX model saved to {out_path}")
            print("Note: XGBoost ONNX contains the model only; use the saved scaler separately.")
            return out_path

        if model_type == "lightgbm":
            onnx_bytes = export_lightgbm_onnx(model, n_features)
            out_path = write_onnx_to_targets(onnx_bytes, target_paths)
            print(f"ONNX model saved to {out_path}")
            print("Note: LightGBM ONNX contains the model only; use the saved scaler separately.")
            return out_path

        if model_type == "sklearn_hgb":
            onnx_bytes = export_sklearn_regressor_onnx(model, n_features)
            out_path = write_onnx_to_targets(onnx_bytes, target_paths)
            print(f"ONNX model saved to {out_path}")
            print("Note: Sklearn HGB ONNX contains the model only; use the saved scaler separately.")
            return out_path

        if model_type == "voting":
            print(
                f"Standalone ONNX export skipped for {symbol} / voting: "
                "ensemble inference is executed in MT5 by averaging the four base ONNX regressors."
            )
            return None

        if convert_sklearn is None or FloatTensorType is None:
            print(f"ONNX export skipped for {model_type}: skl2onnx is not installed.")
            return None

        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", model),
        ])
        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        out_path = write_onnx_to_targets(onnx_model.SerializeToString(), target_paths)
        print(f"ONNX model saved to {out_path}")
        return out_path
    except Exception as exc:
        print(f"ONNX export skipped for {symbol} / {model_type}: {exc}")
        return None


def is_catboost_memory_error(exc):
    return isinstance(exc, (MemoryError, CatBoostError)) and "bad allocation" in str(exc).lower()


def model_artifacts_exist(symbol, timeframe, model_type, variant=None):
    model_file = config.model_path(symbol, timeframe, model_type, variant=variant)
    params_file = config.params_path(symbol, timeframe, model_type, variant=variant)
    if not model_file.exists() or not params_file.exists():
        return False

    if model_type in {"xgboost", "lightgbm", "catboost", "sklearn_hgb"}:
        if not config.onnx_model_path(symbol, timeframe, model_type, variant=variant).exists():
            return False

    return True


def export_scaler_params(symbol, timeframe, scaler, feature_names, variant=None):
    scaler_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    )
    target_paths = [
        config.scaler_params_project_path(symbol, timeframe, variant=variant),
        config.scaler_params_terminal_path(symbol, timeframe, variant=variant),
    ]
    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_df.to_csv(path, index=False)
    print(f"Scaler parameters saved to {target_paths[0]}")


def estimate_model_thresholds(predictions, targets):
    pred_series = pd.Series(predictions, dtype=float)
    target_series = pd.Series(targets, dtype=float).reindex(pred_series.index)
    positive_preds = pred_series[pred_series > 0]
    negative_preds = pred_series[pred_series < 0]

    best = None
    for quantile in config.THRESHOLD_TUNING_QUANTILES:
        buy_threshold = float(positive_preds.quantile(quantile)) if not positive_preds.empty else float("inf")
        sell_threshold = float(negative_preds.quantile(1 - quantile)) if not negative_preds.empty else float("-inf")

        signals = pd.Series(0, index=pred_series.index, dtype=int)
        signals.loc[pred_series >= buy_threshold] = 1
        signals.loc[pred_series <= sell_threshold] = -1
        trade_count = int((signals != 0).sum())
        if trade_count < config.THRESHOLD_MIN_TRAIN_TRADES:
            continue

        strategy_returns = signals * target_series
        total_return_proxy = float(strategy_returns.sum())
        expectancy = float(strategy_returns[signals != 0].mean()) if trade_count > 0 else 0.0

        candidate = {
            "quantile": float(quantile),
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "trade_count": trade_count,
            "total_return_proxy": total_return_proxy,
            "expectancy": expectancy,
        }
        if (
            best is None
            or candidate["total_return_proxy"] > best["total_return_proxy"]
            or (
                candidate["total_return_proxy"] == best["total_return_proxy"]
                and candidate["trade_count"] > best["trade_count"]
            )
        ):
            best = candidate

    if best is None:
        trade_count = int(((pred_series >= config.THRESHOLD_BUY) | (pred_series <= config.THRESHOLD_SELL)).sum())
        best = {
            "quantile": None,
            "buy_threshold": float(config.THRESHOLD_BUY),
            "sell_threshold": float(config.THRESHOLD_SELL),
            "trade_count": trade_count,
            "total_return_proxy": 0.0,
            "expectancy": 0.0,
        }
    return best


def export_thresholds(symbol, timeframe, model_type, threshold_info, variant=None):
    threshold_df = pd.DataFrame(
        [
            {
                "MODEL_NAME": config.runtime_model_name(model_type),
                "BUY_THRESHOLD": threshold_info["buy_threshold"],
                "SELL_THRESHOLD": threshold_info["sell_threshold"],
                "SOURCE_QUANTILE": threshold_info["quantile"],
                "TRAIN_TRADE_COUNT": threshold_info["trade_count"],
                "TRAIN_RETURN_PROXY": threshold_info["total_return_proxy"],
                "TRAIN_EXPECTANCY": threshold_info["expectancy"],
            }
        ]
    )
    target_paths = [
        config.thresholds_path(symbol, timeframe, model_type, variant=variant),
        config.terminal_thresholds_path(symbol, timeframe, model_type, variant=variant),
    ]
    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        threshold_df.to_csv(path, index=False)
    print(
        f"Thresholds saved to {target_paths[0]} | "
        f"buy={threshold_info['buy_threshold']:.6f} | sell={threshold_info['sell_threshold']:.6f}"
    )


def apply_event_target_suppression(
    y_series,
    metadata_df,
    symbol,
    model_type,
    suppression_enabled=False,
    suppression_factor=None,
):
    adjusted = y_series.astype(float).copy()
    summary = {
        "enabled": bool(suppression_enabled),
        "model_type": model_type,
        "factor": None,
        "suppressed_rows": 0,
        "suppressed_events": [],
    }
    if not suppression_enabled:
        return adjusted, summary

    factor = config.MODEL_EVENT_TARGET_FACTOR if suppression_factor is None else float(suppression_factor)
    suppressed_events = config.model_event_blacklist(symbol, model_type)
    summary["factor"] = factor
    summary["suppressed_events"] = suppressed_events
    if not suppressed_events or metadata_df is None or metadata_df.empty or "event_name" not in metadata_df.columns:
        return adjusted, summary

    normalized = {name.casefold() for name in suppressed_events}
    event_names = metadata_df["event_name"].fillna("").astype(str).str.strip()
    mask = event_names.str.casefold().isin(normalized)
    if not mask.any():
        return adjusted, summary

    adjusted.loc[mask] = adjusted.loc[mask] * factor
    summary["suppressed_rows"] = int(mask.sum())
    return adjusted, summary

def train_model_for_symbol(
    symbol,
    model_types=None,
    skip_existing=False,
    full_data=False,
    train_all_data=False,
    data_path=None,
    news_path=None,
    export_runtime_artifacts_enabled=True,
    suppress_blacklist_events=False,
    suppression_factor=None,
):
    artifact_variant = "full" if full_data else None
    data_variant = artifact_variant
    if full_data:
        mode_label = "full-data"
    elif train_all_data:
        mode_label = "train-all-data"
    else:
        mode_label = "split-data"
    print(f"Starting Training Pipeline for {symbol} ({mode_label})...")

    if data_path:
        timeframe = config.infer_timeframe_from_filename(data_path)
        df = load_data_from_csv(data_path)
    else:
        timeframe = config.resolve_pair_timeframe(symbol, variant=data_variant)
        try:
            df = load_pair_data(symbol, timeframe=timeframe, variant=data_variant)
        except FileNotFoundError as e:
            print(f"Skipping {symbol}: {e}")
            return

    resolved_news_path = news_path or config.news_data_path(data_variant)
    news_df = load_news_data(path=resolved_news_path)
    df_features = calculate_technical_features(df)
    news_feature_df, event_metadata_df = build_news_feature_frame(df_features.index, symbol, news_df=news_df)
    df_enriched = df_features.join(news_feature_df, how="left")
    df_enriched[config.NEWS_FEATURE_COLUMNS] = df_enriched[config.NEWS_FEATURE_COLUMNS].fillna(0.0)

    df_target = build_target(df_enriched, horizon=config.PREDICTION_HORIZON, target_type=config.TARGET_TYPE)
    event_metadata_df = event_metadata_df.reindex(df_target.index)

    event_mask = df_target["news_event_present"].fillna(0) > 0
    df_target = df_target.loc[event_mask].copy()
    event_metadata_df = event_metadata_df.loc[df_target.index].copy()
    df_target.dropna(inplace=True)
    event_metadata_df = event_metadata_df.loc[df_target.index]

    if df_target.empty:
        raise ValueError(f"No usable training rows available for {symbol} after event-window filtering.")

    if full_data or train_all_data:
        train_df = df_target.copy()
        test_df = df_target.iloc[0:0].copy()
    elif config.TRAIN_TEST_SPLIT_MODE == "date":
        train_df = df_target[df_target.index < config.TRAIN_END_DATE]
        test_df = df_target[df_target.index >= config.TRAIN_END_DATE]
    else:
        split_idx = int(len(df_target) * (1 - config.TEST_SIZE))
        train_df = df_target.iloc[:split_idx]
        test_df = df_target.iloc[split_idx:]

    train_metadata_df = event_metadata_df.loc[train_df.index].copy()
    test_metadata_df = event_metadata_df.loc[test_df.index].copy()

    features = [col for col in config.FEATURE_COLUMNS if col in train_df.columns]
    missing_features = [col for col in config.FEATURE_COLUMNS if col not in train_df.columns]
    if missing_features:
        print(f"Warning: missing features for {symbol}: {missing_features}")
    if not features:
        raise ValueError(f"No usable features found for {symbol}.")
    if train_df.empty:
        raise ValueError(f"No training data available for {symbol} after event-window filtering.")
    if not full_data and not train_all_data and test_df.empty:
        raise ValueError(f"No test data available for {symbol} after the configured split.")

    X_train = train_df[features]
    y_train = train_df["target"]
    X_test = test_df[features] if not (full_data or train_all_data) else None

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = None if (full_data or train_all_data) else pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    joblib.dump(scaler, config.scaler_path(symbol, timeframe, variant=artifact_variant))
    with open(config.feature_schema_path(symbol, timeframe, variant=artifact_variant), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    export_scaler_params(symbol, timeframe, scaler, features, variant=artifact_variant)

    selected_model_types = model_types or config.MODEL_TYPES

    for model_type in selected_model_types:
        print(f"--- Processing Model: {model_type} ---")
        if skip_existing and model_artifacts_exist(symbol, timeframe, model_type, variant=artifact_variant):
            print(f"Skipping {symbol} / {model_type}: artifacts already exist.")
            continue

        adjusted_y_train, suppression_summary = apply_event_target_suppression(
            y_train,
            train_metadata_df,
            symbol,
            model_type,
            suppression_enabled=suppress_blacklist_events,
            suppression_factor=suppression_factor,
        )
        if suppression_summary["enabled"]:
            print(
                f"Applied model event suppression for {model_type}: "
                f"rows={suppression_summary['suppressed_rows']} | "
                f"factor={suppression_summary['factor']} | "
                f"events={len(suppression_summary['suppressed_events'])}"
            )

        if model_type == "voting":
            estimators = []
            for m in ["xgboost", "lightgbm", "catboost", "sklearn_hgb"]:
                param_path = config.params_path(symbol, timeframe, m, variant=artifact_variant)
                if not param_path.exists():
                    print(f"Skipping VotingRegressor: {m} params not found.")
                    estimators = []
                    break
                with open(param_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                estimators.append((m, build_model(m, params)))

            if not estimators:
                continue

            model = VotingRegressor(estimators=estimators)
            best_params = {}
            with open(config.params_path(symbol, timeframe, model_type, variant=artifact_variant), "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2, sort_keys=True)
        else:
            print(f"Running Hyperparameter Optimization for {model_type}...")
            best_params = optimize_hyperparameters(
                X_train_scaled,
                adjusted_y_train,
                model_type=model_type,
                symbol=symbol,
                timeframe=timeframe,
                variant=artifact_variant,
            )
            with open(config.params_path(symbol, timeframe, model_type, variant=artifact_variant), "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2, sort_keys=True)

            model = build_model(model_type, best_params)

        print(f"Training final {model_type} model...")
        try:
            model.fit(X_train_scaled, adjusted_y_train)
        except Exception as exc:
            if model_type == "catboost" and is_catboost_memory_error(exc):
                print(
                    f"Skipping final CatBoost training for {symbol} due to memory pressure: {exc}"
                )
                continue
            raise

        model_file = config.model_path(symbol, timeframe, model_type, variant=artifact_variant)
        joblib.dump(model, model_file)
        print(f"Model saved to {model_file}")

        train_predictions = pd.Series(model.predict(X_train_scaled), index=X_train_scaled.index, dtype=float)
        threshold_info = estimate_model_thresholds(train_predictions, adjusted_y_train)
        export_thresholds(symbol, timeframe, model_type, threshold_info, variant=artifact_variant)

        export_onnx_model(symbol, timeframe, scaler, model, features, model_type, variant=artifact_variant)

    if full_data:
        train_df.to_csv(config.processed_target_path(symbol, timeframe, variant=artifact_variant))
        X_train_scaled.to_csv(config.processed_train_features_path(symbol, timeframe, variant=artifact_variant))
        if export_runtime_artifacts_enabled:
            export_runtime_artifacts(
                symbol,
                timeframe,
                X_train_scaled[features],
                train_metadata_df,
                variant=artifact_variant,
            )
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_mode": "full",
            "features": features,
            "train_end_date": None,
            "train_rows": int(len(train_df)),
            "test_rows": 0,
            "train_event_rows": int(len(train_metadata_df)),
            "test_event_rows": 0,
            "entry_delay_bars": int(config.ENTRY_DELAY_BARS),
            "prediction_horizon": int(config.PREDICTION_HORIZON),
            "target_definition": (
                "return from close[t + entry_delay_bars] "
                "to close[t + entry_delay_bars + prediction_horizon]"
            ),
            "news_window_before_hours": config.NEWS_WINDOW_BEFORE_HOURS,
            "news_window_after_hours": config.NEWS_WINDOW_AFTER_HOURS,
            "source_dataset_dir": str(config.FULL_OHLCV_DATA_DIR),
            "source_news_file": str(resolved_news_path),
            "runtime_events_exported": bool(export_runtime_artifacts_enabled),
            "model_event_suppression_enabled": bool(suppress_blacklist_events),
            "model_event_target_factor": float(config.MODEL_EVENT_TARGET_FACTOR if suppression_factor is None else suppression_factor),
            "model_event_blacklists": config.MODEL_EVENT_BLACKLISTS,
            "active_symbol_event_blacklists": config.MODEL_EVENT_BLACKLISTS.get(symbol, {}),
        }
        with open(config.metadata_path(symbol, timeframe, variant=artifact_variant), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Prepared full-data training artifacts for {symbol} ({len(train_df)} rows).")
        return

    if train_all_data:
        train_df.to_csv(config.processed_target_path(symbol, timeframe))
        X_train_scaled.to_csv(config.processed_train_features_path(symbol, timeframe))
        if export_runtime_artifacts_enabled:
            export_runtime_artifacts(symbol, timeframe, X_train_scaled[features], train_metadata_df)
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data_mode": "train_all_standard_assets",
            "features": features,
            "train_end_date": None,
            "train_rows": int(len(train_df)),
            "test_rows": 0,
            "train_event_rows": int(len(train_metadata_df)),
            "test_event_rows": 0,
            "entry_delay_bars": int(config.ENTRY_DELAY_BARS),
            "prediction_horizon": int(config.PREDICTION_HORIZON),
            "target_definition": (
                "return from close[t + entry_delay_bars] "
                "to close[t + entry_delay_bars + prediction_horizon]"
            ),
            "news_window_before_hours": config.NEWS_WINDOW_BEFORE_HOURS,
            "news_window_after_hours": config.NEWS_WINDOW_AFTER_HOURS,
            "source_dataset_file": str(data_path) if data_path else str(config.resolve_input_data_path(symbol, timeframe, variant=data_variant)),
            "source_news_file": str(resolved_news_path),
            "runtime_events_exported": bool(export_runtime_artifacts_enabled),
            "model_event_suppression_enabled": bool(suppress_blacklist_events),
            "model_event_target_factor": float(config.MODEL_EVENT_TARGET_FACTOR if suppression_factor is None else suppression_factor),
            "model_event_blacklists": config.MODEL_EVENT_BLACKLISTS,
            "active_symbol_event_blacklists": config.MODEL_EVENT_BLACKLISTS.get(symbol, {}),
        }
        with open(config.metadata_path(symbol, timeframe), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Prepared standard-asset full-train artifacts for {symbol} ({len(train_df)} rows).")
        return

    test_df.to_csv(config.processed_target_path(symbol, timeframe))
    X_test_scaled.to_csv(config.processed_test_features_path(symbol, timeframe))
    export_runtime_artifacts(symbol, timeframe, X_test_scaled[features], test_metadata_df)

    with open(config.metadata_path(symbol, timeframe), "w", encoding="utf-8") as f:
        json.dump({
            "symbol": symbol,
            "timeframe": timeframe,
            "data_mode": "split",
            "features": features,
            "train_end_date": config.TRAIN_END_DATE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_event_rows": int(len(train_metadata_df)),
            "test_event_rows": int(len(test_metadata_df)),
            "entry_delay_bars": int(config.ENTRY_DELAY_BARS),
            "prediction_horizon": int(config.PREDICTION_HORIZON),
            "target_definition": (
                "return from close[t + entry_delay_bars] "
                "to close[t + entry_delay_bars + prediction_horizon]"
            ),
            "news_window_before_hours": config.NEWS_WINDOW_BEFORE_HOURS,
            "news_window_after_hours": config.NEWS_WINDOW_AFTER_HOURS,
            "source_news_file": str(resolved_news_path),
            "model_event_suppression_enabled": bool(suppress_blacklist_events),
            "model_event_target_factor": float(config.MODEL_EVENT_TARGET_FACTOR if suppression_factor is None else suppression_factor),
            "model_event_blacklists": config.MODEL_EVENT_BLACKLISTS,
            "active_symbol_event_blacklists": config.MODEL_EVENT_BLACKLISTS.get(symbol, {}),
        }, f, indent=2)
    print(f"Prepared held-out test artifacts for {symbol} ({len(test_df)} rows).")


def train_all_models(full_data=False):
    config.ensure_directories(variant="full" if full_data else None)
    if full_data:
        prepare_full_training_sources()
    for symbol in config.training_symbols(variant="full" if full_data else None):
        train_model_for_symbol(symbol, full_data=full_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train regression models for all available symbols or a selected subset."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(config.training_symbols()),
        help="Symbols to process. Default: all symbols detected in dataset.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(config.MODEL_TYPES),
        help="Model types to process. Default: all configured model families.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose artifacts already exist, useful for resuming after failure.",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Train on 100%% of the available event-filtered data and write outputs to the isolated full-data folders.",
    )
    parser.add_argument(
        "--train-all-data",
        action="store_true",
        help="Train on 100%% of the available event-filtered data and write outputs to the standard Regression_Assets folders.",
    )
    parser.add_argument(
        "--data-csv",
        help="Custom OHLCV/indicator CSV to train from.",
    )
    parser.add_argument(
        "--news-csv",
        help="Custom news CSV to train from.",
    )
    parser.add_argument(
        "--no-runtime-export",
        action="store_true",
        help="Do not export runtime_events/runtime_event_details artifacts.",
    )
    parser.add_argument(
        "--suppress-blacklist-events",
        action="store_true",
        help="Push model-specific blacklist events toward hold by shrinking their training targets.",
    )
    parser.add_argument(
        "--blacklist-target-factor",
        type=float,
        default=config.MODEL_EVENT_TARGET_FACTOR,
        help="Multiplier applied to blacklist-event targets during training. Default: 0.0 (force hold).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full_data and args.train_all_data:
        raise ValueError("--full-data and --train-all-data cannot be used together.")
    if args.data_csv and len(args.symbols) != 1:
        raise ValueError("--data-csv requires exactly one symbol in --symbols.")
    config.ensure_directories(variant="full" if args.full_data else None)
    if args.full_data:
        prepare_full_training_sources()
    for symbol in args.symbols:
        train_model_for_symbol(
            symbol,
            model_types=args.models,
            skip_existing=args.skip_existing,
            full_data=args.full_data,
            train_all_data=args.train_all_data,
            data_path=args.data_csv,
            news_path=args.news_csv,
            export_runtime_artifacts_enabled=not args.no_runtime_export,
            suppress_blacklist_events=args.suppress_blacklist_events,
            suppression_factor=args.blacklist_target_factor,
        )
