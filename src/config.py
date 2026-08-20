from pathlib import Path
import re
from importlib.util import find_spec

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"
NEWS_DATA_PATH = DATASET_DIR / "NEWS_2018.01.01-2026.06.30_ALL_.csv"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MT5_EXPORT_DIR = DATA_DIR / "mt5_export"

FULL_DATA_DIR = BASE_DIR / "data_full"
FULL_DATASET_DIR = BASE_DIR / "dataset_full"
FULL_OHLCV_DATA_DIR = FULL_DATASET_DIR / "ohlcv"
FULL_NEWS_DATA_DIR = FULL_DATASET_DIR / "macro_news"
FULL_RAW_DATA_DIR = FULL_DATA_DIR / "raw"
FULL_PROCESSED_DATA_DIR = FULL_DATA_DIR / "processed"
FULL_MT5_EXPORT_DIR = FULL_DATA_DIR / "mt5_export"

TERMINAL_MQL5_DIR = BASE_DIR.parent.parent
TERMINAL_FILES_DIR = TERMINAL_MQL5_DIR / "Files"
TERMINAL_ASSETS_DIR = TERMINAL_FILES_DIR / "Assets"
TERMINAL_INSTANCE_ID = TERMINAL_MQL5_DIR.parent.name
TESTER_ROOT_DIR = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Tester" / TERMINAL_INSTANCE_ID
COMMON_FILES_DIR = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" / "Common" / "Files"
COMMON_REGRESSION_ASSETS_DIR = COMMON_FILES_DIR / "Regression_Assets"
COMMON_REGRESSION_ASSETS_DIR_FULL = COMMON_FILES_DIR / "Regression_Assets_Full"

MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
PARAMS_DIR = MODELS_DIR / "params"

FULL_MODELS_DIR = BASE_DIR / "models_full"
FULL_TRAINED_MODELS_DIR = FULL_MODELS_DIR / "trained"
FULL_PARAMS_DIR = FULL_MODELS_DIR / "params"

REPORTS_DIR = BASE_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
CHARTS_DIR = REPORTS_DIR / "charts"
BACKTEST_DIR = REPORTS_DIR / "backtest"
TUNING_DIR = REPORTS_DIR / "tuning"

FULL_REPORTS_DIR = BASE_DIR / "reports_full"
FULL_METRICS_DIR = FULL_REPORTS_DIR / "metrics"
FULL_CHARTS_DIR = FULL_REPORTS_DIR / "charts"
FULL_BACKTEST_DIR = FULL_REPORTS_DIR / "backtest"
FULL_TUNING_DIR = FULL_REPORTS_DIR / "tuning"

ASSETS_DIR = BASE_DIR / "Assets"
FULL_ASSETS_DIR = BASE_DIR / "Assets_full"

# Trading universe
DEFAULT_PAIR_SYMBOLS = ("EURUSD", "USDJPY", "EURJPY", "XAUUSD")
DEFAULT_REPORT_SYMBOLS = ("EURUSD", "USDJPY", "EURJPY")
HYBRID_SYMBOL = "HYBRID"

# MT5 Connection Settings
MT5_SYMBOL = "EURUSD"
MT5_TIMEFRAME = "H1" # e.g., M5, M15, H1
MT5_START_DATE = "2018-01-01"
MT5_END_DATE = "2026-06-30"

# Feature Engineering
TECHNICAL_INDICATORS = {
    "sma_periods": [10, 20, 50],
    "ema_periods": [10, 20, 50],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
}

BASE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "hour",
    "day_of_week",
    "rsi",
    "atr",
    "macd_main",
    "macd_signal",
    "bands_upper",
    "bands_lower",
    "stoch_main",
    "stoch_signal",
    "adx_main",
    "adx_plus_di",
    "adx_minus_di",
    "rsi_h4",
    "ma_h4",
    "rsi_m15",
    "atr_m15",
]

DERIVED_FEATURE_COLUMNS = [
    "body",
    "upper_wick",
    "lower_wick",
    "log_return",
    "volatility_20",
]

NEWS_FEATURE_COLUMNS = [
    "news_event_present",
    "news_event_count",
    "news_max_importance",
    "news_nearest_offset_hours",
    "base_news_count",
    "base_max_importance",
    "base_actual_sum",
    "base_forecast_sum",
    "base_previous_sum",
    "base_surprise_actual_forecast_sum",
    "base_surprise_actual_previous_sum",
    "quote_news_count",
    "quote_max_importance",
    "quote_actual_sum",
    "quote_forecast_sum",
    "quote_previous_sum",
    "quote_surprise_actual_forecast_sum",
    "quote_surprise_actual_previous_sum",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS

RUNTIME_METADATA_COLUMNS = [
    "bar_time",
    "event_time",
    "event_name",
    "country_code",
    "importance",
    "news_max_importance",
    "news_event_count",
    "news_nearest_offset_hours",
]

SYMBOL_NEWS_COUNTRY_MAP = {
    "EUR": "EU",
    "USD": "US",
    "GBP": "GB",
    "AUD": "AU",
    "NZD": "NZ",
    "CAD": "CA",
    "CHF": "CH",
    "JPY": "JP",
}

# Target Settings
TARGET_TYPE = "return" # 'return', 'close', 'direction'
PREDICTION_HORIZON = 1 # Steps ahead to predict
ENTRY_DELAY_BARS = 1 # Trade after the feature/event bar has closed.

# Modeling Settings
MODEL_TYPES = ["xgboost", "lightgbm", "catboost", "sklearn_hgb", "voting"]
TRAIN_TEST_SPLIT_MODE = "date" # 'date', 'time_series', 'expanding_window'
TRAIN_END_DATE = "2024-01-01"
TEST_SIZE = 0.2
N_SPLITS_CV = 5 # For TimeSeriesSplit
CATBOOST_THREAD_COUNT = 2
CATBOOST_USED_RAM_LIMIT = "3gb"

# Optuna Hyperparameter Optimization
OPTUNA_N_TRIALS = 50
OPTUNA_METRIC = "rmse" # 'rmse', 'mae', 'profit_factor'

# Economic news window settings
NEWS_WINDOW_BEFORE_HOURS = 0
NEWS_WINDOW_AFTER_HOURS = 2

# Signal Generation
THRESHOLD_BUY = 0.00015  # Minimal expected positive return to buy
THRESHOLD_SELL = -0.00015 # Minimal expected negative return to sell
THRESHOLD_TUNING_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
THRESHOLD_MIN_TRAIN_TRADES = 30

# Model-side event suppression settings.
# Structure:
# {
#   "EURUSD": {"xgboost": [...], "lightgbm": [...], ...},
#   "GBPUSD": {"xgboost": [...], ...},
# }
MODEL_EVENT_BLACKLISTS = {
    "EURUSD": {
        "xgboost": [
            "Retail Sales m/m",
            "ECB Households Loans y/y",
            "S&P Global Construction PMI",
            "Consumer Price Expectations",
            "Trade Balance",
            "Wage Costs y/y",
            "Construction Output m/m",
            "ZEW Economic Sentiment Indicator",
            "CFTC EUR Non-Commercial Net Positions",
        ],
        "lightgbm": [
            "Current Account",
            "S&P Global Services PMI",
            "Industrial Production m/m",
            "CPI y/y",
            "S&P Global Manufacturing PMI",
            "S&P Global Construction PMI",
            "Trade Balance",
            "Consumer Price Expectations",
            "ECB Households Loans y/y",
        ],
        "catboost": [
            "ECB Households Loans y/y",
            "S&P Global Construction PMI",
            "Retail Sales m/m",
            "S&P Global Services PMI",
        ],
        "sklearn_hgb": [
            "ECB Households Loans y/y",
            "S&P Global Manufacturing PMI",
            "Official Reserve Assets",
        ],
        "voting": [
            "ECB Households Loans y/y",
            "S&P Global Construction PMI",
            "Trade Balance",
            "Retail Sales m/m",
            "Consumer Price Expectations",
            "S&P Global Services PMI",
        ],
    },
    "AUDUSD": {
        "xgboost": [
            "RBA Index of Commodity Prices y/y",
            "ANZ Job Advertisements m/m",
            "CPI q/q",
            "Trade Balance",
            "Construction Work Done q/q",
            "Export Price Index q/q",
            "NAB Business Confidence",
            "GDP q/q",
            "Ai Group Industry Index",
            "S&P Global Manufacturing PMI",
        ],
        "lightgbm": [
            "RBA Index of Commodity Prices y/y",
            "NAB Business Confidence",
            "MI Inflation Expectations",
            "S&P Global Services PMI",
            "CPI q/q",
            "Construction Work Done q/q",
            "PPI q/q",
            "Building Approvals m/m",
            "Ai Group Industry Index",
        ],
        "catboost": [
            "RBA Index of Commodity Prices y/y",
            "CPI q/q",
            "Construction Work Done q/q",
            "ANZ Job Advertisements m/m",
            "Employment Change",
            "NAB Business Confidence",
            "Building Approvals m/m",
            "Trade Balance",
            "Export Price Index q/q",
            "Wage Price Index q/q",
            "S&P Global Manufacturing PMI",
            "MI Inflation Expectations",
        ],
        "sklearn_hgb": [
            "S&P Global Manufacturing PMI",
            "RBA Index of Commodity Prices y/y",
            "RBA Housing Credit m/m",
            "S&P Global Services PMI",
            "MI Inflation Expectations",
            "GDP q/q",
            "PPI q/q",
            "NAB Business Confidence",
            "CFTC AUD Non-Commercial Net Positions",
            "CPI q/q",
            "Retail Sales m/m",
        ],
        "voting": [
            "RBA Index of Commodity Prices y/y",
            "MI Inflation Expectations",
            "CPI q/q",
            "Construction Work Done q/q",
            "NAB Business Confidence",
            "Trade Balance",
            "Export Price Index q/q",
            "Employment Change",
            "S&P Global Services PMI",
            "PPI q/q",
            "ANZ Job Advertisements m/m",
        ],
    },
    "GBPUSD": {
        "xgboost": [
            "CFTC GBP Non-Commercial Net Positions",
            "CPI y/y",
            "S&P Global/CIPS Construction PMI",
            "5-Year Treasury Gilt Auction",
            "HPI y/y",
            "BoE Consumer Credit m/m",
            "Labour Productivity q/q",
            "Manufacturing Production m/m",
            "BoE Interest Rate Decision",
            "S&P Global/CIPS Manufacturing PMI",
            "Nationwide HPI m/m",
            "Trade Balance",
            "Public Sector Net Borrowing",
            "BoE Inflation Expectations",
            "RICS House Price Balance",
            "Halifax HPI m/m",
            "Public Sector Net Cash Requirement",
            "30-Year Treasury Gilt Auction",
            "GDP m/m",
        ],
        "lightgbm": [
            "CFTC GBP Non-Commercial Net Positions",
            "Manufacturing Production m/m",
            "30-Year Treasury Gilt Auction",
            "CPI y/y",
            "5-Year Treasury Gilt Auction",
            "BoE Interest Rate Decision",
            "HPI y/y",
            "RICS House Price Balance",
            "Trade Balance",
            "BoE Inflation Expectations",
            "Public Sector Net Borrowing",
            "Labour Productivity q/q",
            "BoE Consumer Credit m/m",
            "Public Sector Net Cash Requirement",
            "CPIH m/m",
            "Current Account",
            "10-Year Treasury Gilt Auction",
            "Nationwide HPI m/m",
        ],
        "catboost": [
            "CFTC GBP Non-Commercial Net Positions",
            "HPI y/y",
            "Nationwide HPI m/m",
            "S&P Global/CIPS Construction PMI",
            "GDP m/m",
            "BoE Interest Rate Decision",
            "GDP q/q",
            "S&P Global/CIPS Manufacturing PMI",
            "30-Year Treasury Gilt Auction",
            "New Car Registrations m/m",
            "Labour Productivity q/q",
            "Trade Balance",
            "Public Sector Net Borrowing",
            "RICS House Price Balance",
            "BoE Consumer Credit m/m",
            "Public Sector Net Cash Requirement",
            "GfK Consumer Confidence",
        ],
        "sklearn_hgb": [
            "CFTC GBP Non-Commercial Net Positions",
            "New Car Registrations m/m",
            "Nationwide HPI m/m",
            "5-Year Treasury Gilt Auction",
            "BoE Interest Rate Decision",
            "GDP q/q",
            "GDP m/m",
            "Public Sector Net Borrowing",
            "30-Year Treasury Gilt Auction",
            "HPI y/y",
            "BoE Inflation Expectations",
            "Manufacturing Production m/m",
            "CPI y/y",
            "BoE Consumer Credit m/m",
            "S&P Global/CIPS Construction PMI",
            "CPIH m/m",
            "Unemployment Rate",
            "10-Year Treasury Gilt Auction",
            "RICS House Price Balance",
        ],
        "voting": [
            "CFTC GBP Non-Commercial Net Positions",
            "CPI y/y",
            "HPI y/y",
            "GDP m/m",
            "Nationwide HPI m/m",
            "GDP q/q",
            "New Car Registrations m/m",
            "BoE Interest Rate Decision",
            "30-Year Treasury Gilt Auction",
            "S&P Global/CIPS Manufacturing PMI",
            "BoE Inflation Expectations",
            "Labour Productivity q/q",
            "Trade Balance",
            "Public Sector Net Borrowing",
            "10-Year Treasury Gilt Auction",
            "S&P Global/CIPS Construction PMI",
            "Public Sector Net Cash Requirement",
            "Manufacturing Production m/m",
            "BoE Consumer Credit m/m",
        ],
    },
    "NZDUSD": {},
    "USDCAD": {},
    "USDCHF": {},
    "USDJPY": {
        "xgboost": [
            "CFTC JPY Non-Commercial Net Positions",
            "BoJ Trimmed Mean Core CPI y/y",
            "au Jibun Bank Manufacturing PMI",
            "Construction Orders y/y",
            "Consumer Confidence Index",
            "Adjusted Trade Balance",
            "BSI Large Manufacturing",
            "Machine Tool Orders y/y",
            "BoJ Corporate Services Price Index y/y",
            "International Reserves",
            "BoJ Interest Rate Decision",
            "BoJ Bank Lending y/y",
            "au Jibun Bank Services PMI",
            "Tertiary Industry Activity Index m/m",
        ],
        "lightgbm": [
            "CFTC JPY Non-Commercial Net Positions",
            "BoJ Trimmed Mean Core CPI y/y",
            "Adjusted Trade Balance",
            "10-Year JGB Auction",
            "Consumer Confidence Index",
            "Core Machinery Orders m/m",
            "Construction Orders y/y",
            "Labor Cash Earnings y/y",
            "au Jibun Bank Manufacturing PMI",
            "BoJ Monetary Base y/y",
            "Current Account n.s.a.",
            "BSI Large Manufacturing",
            "International Reserves",
            "30-Year JGB Auction",
            "Tertiary Industry Activity Index m/m",
            "Machine Tool Orders y/y",
        ],
        "catboost": [
            "CFTC JPY Non-Commercial Net Positions",
            "Construction Orders y/y",
            "Consumer Confidence Index",
            "au Jibun Bank Manufacturing PMI",
            "10-Year JGB Auction",
            "Adjusted Trade Balance",
            "Machine Tool Orders y/y",
            "30-Year JGB Auction",
            "au Jibun Bank Services PMI",
            "Retail Sales m/m",
            "Core Machinery Orders m/m",
            "Tertiary Industry Activity Index m/m",
            "BoJ L Money Stock y/y",
            "Coincident Index",
            "BSI Large Manufacturing",
            "BoJ Bank Lending y/y",
            "International Reserves",
            "BoJ Interest Rate Decision",
            "BoJ Corporate Goods Price Index m/m",
            "BoJ Corporate Goods Price Index y/y",
        ],
        "sklearn_hgb": [
            "Consumer Confidence Index",
            "BoJ Monetary Base y/y",
            "au Jibun Bank Services PMI",
            "Core Machinery Orders m/m",
            "10-Year JGB Auction",
            "BoJ Corporate Services Price Index y/y",
            "BSI Large Manufacturing",
            "Industrial Production m/m",
            "BoJ Bank Lending y/y",
            "Economy Watchers Index for Current Conditions",
            "BoJ Trimmed Mean Core CPI y/y",
        ],
    },
}
MODEL_EVENT_TARGET_FACTOR = 0.0


def model_event_blacklist(symbol, model_type):
    """Return the configured blacklist for one symbol/model pair."""
    symbol_key = str(symbol).upper()
    model_key = str(model_type).strip()

    symbol_config = MODEL_EVENT_BLACKLISTS.get(symbol_key)
    if isinstance(symbol_config, dict):
        return [str(item).strip() for item in symbol_config.get(model_key, []) if str(item).strip()]

    # Backward-compatible fallback for older flat configs:
    legacy = MODEL_EVENT_BLACKLISTS.get(model_key, [])
    if isinstance(legacy, (list, tuple, set)):
        return [str(item).strip() for item in legacy if str(item).strip()]
    return []

# Hybrid aggregation
HYBRID_WEIGHT_MODE = "inverse_rmse"  # inverse_rmse, equal, manual
PAIR_WEIGHTS = {}


def _variant_root(variant, standard_root, full_root):
    return full_root if variant == "full" else standard_root


def ensure_directories(variant=None):
    """Create the runtime directories used by the pipeline."""
    data_dir = _variant_root(variant, DATA_DIR, FULL_DATA_DIR)
    dataset_dir = _variant_root(variant, DATASET_DIR, FULL_DATASET_DIR)
    raw_data_dir = _variant_root(variant, RAW_DATA_DIR, FULL_RAW_DATA_DIR)
    processed_data_dir = _variant_root(variant, PROCESSED_DATA_DIR, FULL_PROCESSED_DATA_DIR)
    mt5_export_dir = _variant_root(variant, MT5_EXPORT_DIR, FULL_MT5_EXPORT_DIR)
    models_dir = _variant_root(variant, MODELS_DIR, FULL_MODELS_DIR)
    trained_models_dir = _variant_root(variant, TRAINED_MODELS_DIR, FULL_TRAINED_MODELS_DIR)
    params_dir = _variant_root(variant, PARAMS_DIR, FULL_PARAMS_DIR)
    reports_dir = _variant_root(variant, REPORTS_DIR, FULL_REPORTS_DIR)
    metrics_dir = _variant_root(variant, METRICS_DIR, FULL_METRICS_DIR)
    charts_dir = _variant_root(variant, CHARTS_DIR, FULL_CHARTS_DIR)
    backtest_dir = _variant_root(variant, BACKTEST_DIR, FULL_BACKTEST_DIR)
    tuning_dir = _variant_root(variant, TUNING_DIR, FULL_TUNING_DIR)
    assets_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    regression_assets_dir = _variant_root(
        variant,
        COMMON_REGRESSION_ASSETS_DIR,
        COMMON_REGRESSION_ASSETS_DIR_FULL,
    )

    for path in (
        data_dir,
        dataset_dir,
        dataset_dir / "ohlcv",
        dataset_dir / "macro_news",
        raw_data_dir,
        processed_data_dir,
        mt5_export_dir,
        TERMINAL_FILES_DIR,
        models_dir,
        trained_models_dir,
        params_dir,
        reports_dir,
        metrics_dir,
        charts_dir,
        backtest_dir,
        tuning_dir,
        assets_dir,
        regression_assets_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def tester_asset_target_paths(filename):
    return [COMMON_REGRESSION_ASSETS_DIR / filename]


def pair_key(symbol, timeframe=None):
    """Build the canonical file prefix for a pair."""
    tf = timeframe or MT5_TIMEFRAME
    return f"{symbol}_{tf}"


def news_data_path(variant=None):
    if variant == "full":
        return FULL_NEWS_DATA_DIR / NEWS_DATA_PATH.name
    return NEWS_DATA_PATH


def raw_data_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, RAW_DATA_DIR, FULL_RAW_DATA_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}_raw.csv"


def exported_data_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, MT5_EXPORT_DIR, FULL_MT5_EXPORT_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}_Data.csv"


def dataset_data_path(symbol, timeframe=None, variant=None):
    if variant == "full":
        return FULL_OHLCV_DATA_DIR / f"{pair_key(symbol, timeframe)}_Data.csv"
    return DATASET_DIR / f"{pair_key(symbol, timeframe)}_Data.csv"


def discover_available_symbols(variant=None):
    discovered = []
    patterns = []
    dataset_dir = _variant_root(variant, DATASET_DIR, FULL_DATASET_DIR)
    raw_dir = _variant_root(variant, RAW_DATA_DIR, FULL_RAW_DATA_DIR)
    export_dir = _variant_root(variant, MT5_EXPORT_DIR, FULL_MT5_EXPORT_DIR)
    for folder in (dataset_dir, export_dir, raw_dir):
        if folder.exists():
            patterns.extend(sorted(folder.glob("*_Data.csv")))
            patterns.extend(sorted(folder.glob("*_raw.csv")))

    seen = set()
    for path in patterns:
        stem = path.stem
        match = re.match(r"^([A-Z0-9]+)_[A-Z0-9]+_(?:Data|raw)$", stem)
        if not match:
            continue
        symbol = match.group(1)
        if symbol not in seen:
            seen.add(symbol)
            discovered.append(symbol)
    return tuple(discovered)


def training_symbols(variant=None):
    discovered = discover_available_symbols(variant=variant)
    return discovered or DEFAULT_PAIR_SYMBOLS


def report_symbols():
    discovered = training_symbols()
    return tuple(discovered or DEFAULT_REPORT_SYMBOLS)


def resolve_input_data_path(symbol, timeframe=None, variant=None):
    dataset_path = dataset_data_path(symbol, timeframe, variant=variant)
    if dataset_path.exists():
        return dataset_path

    export_path = exported_data_path(symbol, timeframe, variant=variant)
    if export_path.exists():
        return export_path

    raw_path = raw_data_path(symbol, timeframe, variant=variant)
    if raw_path.exists():
        return raw_path

    raw_dir = _variant_root(variant, RAW_DATA_DIR, FULL_RAW_DATA_DIR)
    if raw_dir.exists():
        matches = sorted(raw_dir.glob(f"{symbol}_*_Data.csv"))
        if matches:
            return matches[0]

    return export_path


def infer_timeframe_from_filename(path):
    """Infer the timeframe token from a file name like EURUSD_H1_Data.csv."""
    stem = Path(path).stem
    match = re.match(r"^[A-Z0-9]+_([A-Z0-9]+)_Data$", stem)
    if match:
        return match.group(1)
    match = re.match(r"^[A-Z0-9]+_([A-Z0-9]+)_raw$", stem)
    if match:
        return match.group(1)
    return MT5_TIMEFRAME


def processed_target_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, PROCESSED_DATA_DIR, FULL_PROCESSED_DATA_DIR)
    suffix = "test_data_with_target" if variant != "full" else "full_data_with_target"
    return base_dir / f"{pair_key(symbol, timeframe)}_{suffix}.csv"


def processed_test_features_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, PROCESSED_DATA_DIR, FULL_PROCESSED_DATA_DIR)
    suffix = "X_test_scaled" if variant != "full" else "X_full_scaled"
    return base_dir / f"{pair_key(symbol, timeframe)}_{suffix}.csv"


def processed_train_features_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, PROCESSED_DATA_DIR, FULL_PROCESSED_DATA_DIR)
    suffix = "X_train_scaled" if variant != "full" else "X_full_scaled"
    return base_dir / f"{pair_key(symbol, timeframe)}_{suffix}.csv"


def model_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, TRAINED_MODELS_DIR, FULL_TRAINED_MODELS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_regression_model.pkl"


def runtime_model_name(model_type):
    return "ensemble_regression" if model_type == "voting" else model_type


def onnx_model_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}.onnx"


def terminal_onnx_model_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}.onnx"


def scaler_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, TRAINED_MODELS_DIR, FULL_TRAINED_MODELS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}_scaler.pkl"


def params_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, PARAMS_DIR, FULL_PARAMS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_best_params.json"


def metadata_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, PARAMS_DIR, FULL_PARAMS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}_metadata.json"


def feature_schema_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, PARAMS_DIR, FULL_PARAMS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}_features.json"


def metrics_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, METRICS_DIR, FULL_METRICS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_evaluation_metrics.csv"


def predictions_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, REPORTS_DIR, FULL_REPORTS_DIR)
    suffix = "test_predictions" if variant != "full" else "full_predictions"
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_{suffix}.csv"


def signal_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_signals.csv"


def terminal_signal_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_signals.csv"


def thresholds_path(symbol, timeframe=None, model_type=None, variant=None):
    model_name = runtime_model_name(model_type) if model_type else ""
    mt = f"_{model_name}" if model_name else ""
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_thresholds.csv"


def terminal_thresholds_path(symbol, timeframe=None, model_type=None, variant=None):
    model_name = runtime_model_name(model_type) if model_type else ""
    mt = f"_{model_name}" if model_name else ""
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_thresholds.csv"


def backtest_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, BACKTEST_DIR, FULL_BACKTEST_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_equity_curve.csv"


def tuning_trials_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, TUNING_DIR, FULL_TUNING_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_tuning_trials.csv"


def tuning_summary_path(symbol, timeframe=None, model_type=None, variant=None):
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, TUNING_DIR, FULL_TUNING_DIR)
    return base_dir / f"{pair_key(symbol, timeframe)}{mt}_tuning_summary.json"


def hybrid_predictions_path(timeframe=None, model_type=None, variant=None):
    tf = timeframe or MT5_TIMEFRAME
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, REPORTS_DIR, FULL_REPORTS_DIR)
    suffix = "test_predictions" if variant != "full" else "full_predictions"
    return base_dir / f"{HYBRID_SYMBOL}_{tf}{mt}_{suffix}.csv"


def hybrid_metrics_path(timeframe=None, model_type=None, variant=None):
    tf = timeframe or MT5_TIMEFRAME
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, METRICS_DIR, FULL_METRICS_DIR)
    return base_dir / f"{HYBRID_SYMBOL}_{tf}{mt}_evaluation_metrics.csv"


def hybrid_signal_path(timeframe=None, model_type=None, variant=None):
    tf = timeframe or MT5_TIMEFRAME
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / f"{HYBRID_SYMBOL}_{tf}{mt}_signals.csv"


def terminal_hybrid_signal_path(timeframe=None, model_type=None, variant=None):
    tf = timeframe or MT5_TIMEFRAME
    mt = f"_{model_type}" if model_type else ""
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / f"{HYBRID_SYMBOL}_{tf}{mt}_signals.csv"


def scaler_params_filename(symbol, timeframe=None):
    return f"{pair_key(symbol, timeframe)}_scaler_params.csv"


def scaler_params_project_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / scaler_params_filename(symbol, timeframe)


def scaler_params_terminal_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / scaler_params_filename(symbol, timeframe)


def runtime_events_filename(symbol, timeframe=None):
    return f"{pair_key(symbol, timeframe)}_runtime_events.csv"


def runtime_events_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / runtime_events_filename(symbol, timeframe)


def terminal_runtime_events_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / runtime_events_filename(symbol, timeframe)


def runtime_event_details_filename(symbol, timeframe=None):
    return f"{pair_key(symbol, timeframe)}_runtime_event_details.csv"


def runtime_event_details_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, ASSETS_DIR, FULL_ASSETS_DIR)
    return base_dir / runtime_event_details_filename(symbol, timeframe)


def terminal_runtime_event_details_path(symbol, timeframe=None, variant=None):
    base_dir = _variant_root(variant, COMMON_REGRESSION_ASSETS_DIR, COMMON_REGRESSION_ASSETS_DIR_FULL)
    return base_dir / runtime_event_details_filename(symbol, timeframe)


def hybrid_weights_path(timeframe=None):
    tf = timeframe or MT5_TIMEFRAME
    return PARAMS_DIR / f"{HYBRID_SYMBOL}_{tf}_weights.json"


def is_installed(module_name):
    return find_spec(module_name) is not None


def resolve_model_type(preferred=None):
    """
    Check if a specific model type is available.
    """
    preferred = (preferred or "sklearn_hgb").lower()
    
    if preferred == "voting":
        return "voting"

    available = {
        "xgboost": is_installed("xgboost"),
        "lightgbm": is_installed("lightgbm"),
        "catboost": is_installed("catboost"),
    }

    if preferred in available and available[preferred]:
        return preferred

    if preferred == "sklearn_hgb":
        return "sklearn_hgb"

    print(f"Warning: {preferred} is not installed.")
    return preferred


def resolve_pair_timeframe(symbol, variant=None):
    """Resolve a pair timeframe from saved metadata or available data files."""
    # Check saved metadata first.
    params_dir = _variant_root(variant, PARAMS_DIR, FULL_PARAMS_DIR)
    if params_dir.exists():
        for path in params_dir.glob(f"{symbol}_*_metadata.json"):
            try:
                import json
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tf = data.get("timeframe")
                if tf:
                    return tf
            except Exception:
                pass

    # Then infer from raw/exported data filenames.
    candidates = []
    dataset_dir = _variant_root(variant, DATASET_DIR, FULL_DATASET_DIR)
    raw_dir = _variant_root(variant, RAW_DATA_DIR, FULL_RAW_DATA_DIR)
    export_dir = _variant_root(variant, MT5_EXPORT_DIR, FULL_MT5_EXPORT_DIR)
    if dataset_dir.exists():
        candidates.extend(dataset_dir.glob(f"{symbol}_*_Data.csv"))
    if raw_dir.exists():
        candidates.extend(raw_dir.glob(f"{symbol}_*_Data.csv"))
        candidates.extend(raw_dir.glob(f"{symbol}_*_raw.csv"))
    if export_dir.exists():
        candidates.extend(export_dir.glob(f"{symbol}_*_Data.csv"))

    if candidates:
        return infer_timeframe_from_filename(candidates[0])

    return MT5_TIMEFRAME
