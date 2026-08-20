import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import config

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None

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


def _build_model(model_type, params):
    if model_type == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install it or switch MODEL_TYPE.")
        return XGBRegressor(**params, random_state=42)
    if model_type == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed. Install it or switch MODEL_TYPE.")
        return LGBMRegressor(**params, random_state=42)
    if model_type == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed. Install it or switch MODEL_TYPE.")
        return CatBoostRegressor(
            **params,
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
        return HistGradientBoostingRegressor(**params, random_state=42)
    raise ValueError(f"Unsupported model type for optimization: {model_type}")


def _score_predictions(y_true, preds):
    if config.OPTUNA_METRIC == "mae":
        return mean_absolute_error(y_true, preds)
    return np.sqrt(mean_squared_error(y_true, preds))


def _is_catboost_memory_error(exc):
    return isinstance(exc, (MemoryError, CatBoostError)) and "bad allocation" in str(exc).lower()


def _evaluate_params(X_train, y_train, model_type, params):
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS_CV)
    scores = []

    for fold_idx, (train_index, test_index) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_te = X_train.iloc[train_index], X_train.iloc[test_index]
        y_tr, y_te = y_train.iloc[train_index], y_train.iloc[test_index]

        model = _build_model(model_type, params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        scores.append(
            {
                "fold": fold_idx,
                "score": float(_score_predictions(y_te, preds)),
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(test_index)),
            }
        )

    mean_score = float(np.mean([item["score"] for item in scores]))
    return mean_score, scores


def _write_tuning_logs(symbol, timeframe, model_type, tuning_method, records, best_params, best_score, variant=None):
    trials_path = config.tuning_trials_path(symbol, timeframe, model_type, variant=variant)
    summary_path = config.tuning_summary_path(symbol, timeframe, model_type, variant=variant)

    trials_df = pd.DataFrame(records)
    trials_df.to_csv(trials_path, index=False)

    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "model_type": model_type,
        "tuning_method": tuning_method,
        "metric": config.OPTUNA_METRIC,
        "n_trials_logged": int(len(records)),
        "cv_splits": int(config.N_SPLITS_CV),
        "best_score": float(best_score),
        "best_params": best_params,
        "trials_path": str(trials_path),
    }

    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Tuning trials saved to {trials_path}")
    print(f"Tuning summary saved to {summary_path}")


def _fallback_candidates(model_type):
    if model_type == "sklearn_hgb":
        return [
            {
                "max_iter": 150,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
            },
            {
                "max_iter": 250,
                "learning_rate": 0.05,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
            },
            {
                "max_iter": 300,
                "learning_rate": 0.03,
                "max_depth": 7,
                "min_samples_leaf": 30,
                "l2_regularization": 0.2,
            },
        ]

    if model_type == "xgboost":
        return [
            {
                "n_estimators": 150,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ]

    if model_type == "lightgbm":
        return [
            {
                "n_estimators": 150,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ]

    if model_type == "catboost":
        return [
            {
                "iterations": 150,
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
            },
            {
                "iterations": 250,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5.0,
            },
        ]

    raise ValueError(f"Unsupported model type for optimization: {model_type}")


def _grid_fallback(X_train, y_train, model_type, symbol=None, timeframe=None, variant=None):
    candidates = _fallback_candidates(model_type)
    best_params = None
    best_score = float("inf")
    records = []

    for candidate_idx, params in enumerate(candidates, start=1):
        score, fold_scores = _evaluate_params(X_train, y_train, model_type, params)
        record = {
            "trial_number": candidate_idx,
            "tuning_method": "fallback_grid",
            "model_type": model_type,
            "mean_validation_score": float(score),
            "best_score_so_far": float(min(best_score, score)),
        }
        for fold_info in fold_scores:
            fold = fold_info["fold"]
            record[f"fold_{fold}_score"] = fold_info["score"]
            record[f"fold_{fold}_train_rows"] = fold_info["train_rows"]
            record[f"fold_{fold}_validation_rows"] = fold_info["validation_rows"]
        record.update(params)
        records.append(record)
        if score < best_score:
            best_score = score
            best_params = params

    if symbol and timeframe:
        _write_tuning_logs(
            symbol,
            timeframe,
            model_type,
            "fallback_grid",
            records,
            best_params or {},
            best_score,
            variant=variant,
        )

    print(f"Best fallback score: {best_score}")
    print(f"Best fallback params: {best_params}")
    return best_params or {}


def optimize_hyperparameters(X_train, y_train, model_type="xgboost", symbol=None, timeframe=None, variant=None):
    """Optimize hyperparameters using Optuna when available; otherwise use a small time-series grid search."""
    if optuna is None:
        print("Optuna not installed. Using fallback time-series grid search.")
        return _grid_fallback(X_train, y_train, model_type, symbol=symbol, timeframe=timeframe, variant=variant)

    def objective(trial):
        if model_type == "xgboost":
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed. Install it or switch MODEL_TYPE.")
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif model_type == "lightgbm":
            if LGBMRegressor is None:
                raise ImportError("lightgbm is not installed. Install it or switch MODEL_TYPE.")
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        elif model_type == "catboost":
            if CatBoostRegressor is None:
                raise ImportError("catboost is not installed. Install it or switch MODEL_TYPE.")
            param = {
                "iterations": trial.suggest_int("iterations", 100, 350),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        elif model_type == "sklearn_hgb":
            param = {
                "max_iter": trial.suggest_int("max_iter", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 2.0),
            }
        else:
            raise ValueError(f"Unsupported model type for optimization: {model_type}")

        tscv = TimeSeriesSplit(n_splits=config.N_SPLITS_CV)
        scores = []
        fold_details = []
        for fold_idx, (train_index, test_index) in enumerate(tscv.split(X_train), start=1):
            X_tr, X_te = X_train.iloc[train_index], X_train.iloc[test_index]
            y_tr, y_te = y_train.iloc[train_index], y_train.iloc[test_index]

            model = _build_model(model_type, param)
            try:
                model.fit(X_tr, y_tr)
            except Exception as exc:
                if model_type == "catboost" and _is_catboost_memory_error(exc):
                    trial.set_user_attr(
                        "failure_reason",
                        f"catboost_memory_error_fold_{fold_idx}: {str(exc)}",
                    )
                    return float("inf")
                raise
            preds = model.predict(X_te)
            fold_score = float(_score_predictions(y_te, preds))
            scores.append(fold_score)
            fold_details.append(
                {
                    "fold": fold_idx,
                    "score": fold_score,
                    "train_rows": int(len(train_index)),
                    "validation_rows": int(len(test_index)),
                }
            )

        trial.set_user_attr("fold_details", fold_details)

        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, catch=(MemoryError,))

    if symbol and timeframe:
        records = []
        best_so_far = float("inf")
        for trial in study.trials:
            if trial.value is None:
                continue
            best_so_far = min(best_so_far, float(trial.value))
            record = {
                "trial_number": int(trial.number),
                "tuning_method": "optuna",
                "model_type": model_type,
                "mean_validation_score": float(trial.value),
                "best_score_so_far": float(best_so_far),
                "trial_state": str(trial.state),
            }
            for fold_info in trial.user_attrs.get("fold_details", []):
                fold = fold_info["fold"]
                record[f"fold_{fold}_score"] = fold_info["score"]
                record[f"fold_{fold}_train_rows"] = fold_info["train_rows"]
                record[f"fold_{fold}_validation_rows"] = fold_info["validation_rows"]
            record.update(trial.params)
            records.append(record)

        _write_tuning_logs(
            symbol,
            timeframe,
            model_type,
            "optuna",
            records,
            study.best_trial.params,
            float(study.best_trial.value),
            variant=variant,
        )

    print(f"Best trial value: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    return study.best_trial.params
