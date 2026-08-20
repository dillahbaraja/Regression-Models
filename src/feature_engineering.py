import pandas as pd
import numpy as np
import config


def calculate_technical_features(df):
    """Use dataset-provided technical columns and add lightweight derived candle features."""
    df_feat = df.copy()

    for col in config.BASE_FEATURE_COLUMNS:
        if col not in df_feat.columns:
            raise ValueError(f"Missing required base feature column: {col}")

    df_feat['body'] = df_feat['close'] - df_feat['open']
    df_feat['upper_wick'] = df_feat['high'] - df_feat[['open', 'close']].max(axis=1)
    df_feat['lower_wick'] = df_feat[['open', 'close']].min(axis=1) - df_feat['low']
    df_feat['log_return'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
    df_feat['volatility_20'] = df_feat['log_return'].rolling(window=20).std()

    if 'hour' not in df_feat.columns:
        df_feat['hour'] = df_feat.index.hour
    if 'day_of_week' not in df_feat.columns:
        df_feat['day_of_week'] = df_feat.index.dayofweek

    return df_feat

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    pass
