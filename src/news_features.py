from pathlib import Path

import numpy as np
import pandas as pd

import config


NEWS_NUMERIC_COLUMNS = ["Actual", "Forecast", "Previous", "Impact", "Importance"]


def resolve_symbol_country_codes(symbol):
    symbol = symbol.upper()
    base = symbol[:3]
    quote = symbol[3:6]
    try:
        return config.SYMBOL_NEWS_COUNTRY_MAP[base], config.SYMBOL_NEWS_COUNTRY_MAP[quote]
    except KeyError as exc:
        raise ValueError(f"Unsupported symbol for news-country mapping: {symbol}") from exc


def load_news_data(path=None):
    news_path = Path(path or config.NEWS_DATA_PATH)
    if not news_path.exists():
        raise FileNotFoundError(f"News file not found: {news_path}")

    news_df = pd.read_csv(news_path)
    news_df["Datetime"] = pd.to_datetime(news_df["Datetime"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
    news_df = news_df.dropna(subset=["Datetime"]).copy()
    news_df["CountryCode"] = news_df["CountryCode"].astype(str).str.upper().str.strip()

    for col in NEWS_NUMERIC_COLUMNS:
        if col not in news_df.columns:
            news_df[col] = 0.0
        news_df[col] = pd.to_numeric(news_df[col], errors="coerce").fillna(0.0)

    news_df.sort_values("Datetime", inplace=True)
    news_df.reset_index(drop=True, inplace=True)
    return news_df


def _empty_news_frames(index):
    feature_df = pd.DataFrame(0.0, index=index, columns=config.NEWS_FEATURE_COLUMNS)
    feature_df["news_event_present"] = 0
    metadata_df = pd.DataFrame(index=index)
    metadata_df["bar_time"] = index
    metadata_df["event_time"] = pd.NaT
    metadata_df["event_name"] = ""
    metadata_df["country_code"] = ""
    metadata_df["importance"] = 0
    metadata_df["news_max_importance"] = 0
    metadata_df["news_event_count"] = 0
    metadata_df["news_nearest_offset_hours"] = np.nan
    return feature_df, metadata_df


def build_news_feature_frame(index, symbol, news_df=None):
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    feature_df, metadata_df = _empty_news_frames(index)
    if len(index) == 0:
        return feature_df, metadata_df

    base_code, quote_code = resolve_symbol_country_codes(symbol)
    news_df = load_news_data() if news_df is None else news_df
    relevant_news = news_df[news_df["CountryCode"].isin([base_code, quote_code])].copy()
    if relevant_news.empty:
        return feature_df, metadata_df

    bar_times = index.to_numpy()
    before = pd.Timedelta(hours=config.NEWS_WINDOW_BEFORE_HOURS)
    after = pd.Timedelta(hours=config.NEWS_WINDOW_AFTER_HOURS)

    best_abs_offset = np.full(len(index), np.inf, dtype=float)
    best_importance = np.full(len(index), -1, dtype=float)

    for row in relevant_news.itertuples(index=False):
        event_time = pd.Timestamp(row.Datetime)
        start_idx = index.searchsorted(event_time - before, side="left")
        end_idx = index.searchsorted(event_time + after, side="right")
        if start_idx >= end_idx:
            continue

        importance = int(getattr(row, "Importance", 0) or 0)
        actual = float(getattr(row, "Actual", 0.0) or 0.0)
        forecast = float(getattr(row, "Forecast", 0.0) or 0.0)
        previous = float(getattr(row, "Previous", 0.0) or 0.0)
        surprise_af = actual - forecast
        surprise_ap = actual - previous
        country_code = str(getattr(row, "CountryCode", "") or "").upper()
        event_name = str(getattr(row, "Name", "") or "")

        for idx_pos in range(start_idx, end_idx):
            bar_time = pd.Timestamp(bar_times[idx_pos])
            offset_hours = (bar_time - event_time).total_seconds() / 3600.0
            if offset_hours < 0 or offset_hours > config.NEWS_WINDOW_AFTER_HOURS:
                continue

            feature_df.iat[idx_pos, feature_df.columns.get_loc("news_event_present")] = 1
            feature_df.iat[idx_pos, feature_df.columns.get_loc("news_event_count")] += 1
            feature_df.iat[idx_pos, feature_df.columns.get_loc("news_max_importance")] = max(
                feature_df.iat[idx_pos, feature_df.columns.get_loc("news_max_importance")],
                importance,
            )

            abs_offset = abs(offset_hours)
            if (
                abs_offset < best_abs_offset[idx_pos]
                or (
                    np.isclose(abs_offset, best_abs_offset[idx_pos])
                    and importance > best_importance[idx_pos]
                )
            ):
                best_abs_offset[idx_pos] = abs_offset
                best_importance[idx_pos] = importance
                feature_df.iat[idx_pos, feature_df.columns.get_loc("news_nearest_offset_hours")] = offset_hours
                metadata_df.iat[idx_pos, metadata_df.columns.get_loc("event_time")] = event_time
                metadata_df.iat[idx_pos, metadata_df.columns.get_loc("event_name")] = event_name
                metadata_df.iat[idx_pos, metadata_df.columns.get_loc("country_code")] = country_code
                metadata_df.iat[idx_pos, metadata_df.columns.get_loc("importance")] = importance
                metadata_df.iat[idx_pos, metadata_df.columns.get_loc("news_nearest_offset_hours")] = offset_hours

            prefix = "base" if country_code == base_code else "quote"
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_news_count")] += 1
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_max_importance")] = max(
                feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_max_importance")],
                importance,
            )
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_actual_sum")] += actual
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_forecast_sum")] += forecast
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_previous_sum")] += previous
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_surprise_actual_forecast_sum")] += surprise_af
            feature_df.iat[idx_pos, feature_df.columns.get_loc(f"{prefix}_surprise_actual_previous_sum")] += surprise_ap

    metadata_df["news_max_importance"] = feature_df["news_max_importance"].astype(int)
    metadata_df["news_event_count"] = feature_df["news_event_count"].astype(int)
    return feature_df, metadata_df
