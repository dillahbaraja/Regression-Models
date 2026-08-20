import pandas as pd
import numpy as np
import config

def build_target(df, horizon=1, target_type="return", entry_delay_bars=None):
    """
    Build the target variable to predict.
    The default execution assumption is bar-close execution: features from bar t
    are acted on after that bar has closed, so the tradable return starts at
    t + ENTRY_DELAY_BARS rather than at t.
    """
    df_target = df.copy()
    entry_delay = config.ENTRY_DELAY_BARS if entry_delay_bars is None else entry_delay_bars
    entry_close = df_target['close'].shift(-entry_delay)
    exit_close = df_target['close'].shift(-(entry_delay + horizon))
    
    if target_type == "return":
        df_target['target'] = (exit_close - entry_close) / entry_close
    elif target_type == "log_return":
        df_target['target'] = np.log(exit_close / entry_close)
    elif target_type == "close":
        df_target['target'] = exit_close
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
        
    # Drop NaNs that appear due to shifting at the end of the dataset
    df_target.dropna(subset=['target'], inplace=True)
    return df_target

if __name__ == "__main__":
    pass
