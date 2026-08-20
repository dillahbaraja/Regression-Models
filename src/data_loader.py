import pandas as pd
from datetime import datetime
import os
import config

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - optional dependency
    mt5 = None


STANDARD_COLUMN_MAP = {
    "Time": "time",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "tick_volume",
    "Hour": "hour",
    "DayOfWeek": "day_of_week",
    "RSI": "rsi",
    "ATR": "atr",
    "MACD_Main": "macd_main",
    "MACD_Signal": "macd_signal",
    "Bands_Upper": "bands_upper",
    "Bands_Lower": "bands_lower",
    "Stoch_Main": "stoch_main",
    "Stoch_Signal": "stoch_signal",
    "ADX_Main": "adx_main",
    "ADX_PlusDI": "adx_plus_di",
    "ADX_MinusDI": "adx_minus_di",
    "RSI_H4": "rsi_h4",
    "MA_H4": "ma_h4",
    "RSI_M15": "rsi_m15",
    "ATR_M15": "atr_m15",
    "DATETIME": "time",
    "datetime": "time",
}


def _normalize_exported_columns(df):
    rename_map = {col: STANDARD_COLUMN_MAP[col] for col in df.columns if col in STANDARD_COLUMN_MAP}
    normalized = df.rename(columns=rename_map)

    if "time" not in normalized.columns:
        raise ValueError("CSV must contain a time/DATETIME/Time column.")

    normalized["time"] = pd.to_datetime(normalized["time"])
    normalized.set_index("time", inplace=True)
    normalized.sort_index(inplace=True)

    for col in ("open", "high", "low", "close"):
        if col not in normalized.columns:
            raise ValueError(f"CSV missing required OHLC column: {col}")

    return normalized

def load_data_from_csv(filepath):
    """Load data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)

    if "time" in df.columns or "DATETIME" in df.columns or "Time" in df.columns:
        return _normalize_exported_columns(df)

    raise ValueError(f"Unsupported CSV schema in {filepath}")


def load_pair_data(symbol, timeframe=None, variant=None):
    """Load the best available data source for a symbol."""
    filepath = config.resolve_input_data_path(symbol, timeframe, variant=variant)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return load_data_from_csv(filepath)

def fetch_data_from_mt5(symbol, timeframe_str, start_date_str, end_date_str):
    """Fetch data directly from MetaTrader 5."""
    if mt5 is None:
        raise ImportError("MetaTrader5 package is not installed. Use CSV input or install the package.")
    if not mt5.initialize():
        raise Exception(f"MT5 initialization failed, error code: {mt5.last_error()}")
    
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    tf = timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M15)
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data retrieved for {symbol} on {timeframe_str}")
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    try:
        df = fetch_data_from_mt5(config.MT5_SYMBOL, config.MT5_TIMEFRAME, config.MT5_START_DATE, config.MT5_END_DATE)
        df.to_csv(config.RAW_DATA_DIR / f"{config.MT5_SYMBOL}_{config.MT5_TIMEFRAME}_raw.csv")
        print(f"Data fetched and saved. Shape: {df.shape}")
    except Exception as e:
        print(f"Error fetching MT5 data: {e}. You can place a CSV manually in data/raw.")
