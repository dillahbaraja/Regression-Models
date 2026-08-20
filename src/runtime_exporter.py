import pandas as pd

import config


def _sanitize_csv_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace('"', "", regex=False)
        .str.replace(",", ";", regex=False)
        .str.strip()
    )


def _write_frame_to_targets(df, target_paths):
    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def export_runtime_artifacts(symbol, timeframe, scaled_feature_df, metadata_df, variant=None):
    if scaled_feature_df.empty or metadata_df.empty:
        return

    aligned_metadata = metadata_df.loc[scaled_feature_df.index].copy()
    aligned_metadata["bar_time"] = pd.to_datetime(aligned_metadata["bar_time"])
    aligned_metadata["event_time"] = pd.to_datetime(aligned_metadata["event_time"], errors="coerce")

    runtime_df = pd.DataFrame()
    runtime_df["BAR_TIME"] = aligned_metadata["bar_time"].dt.strftime("%Y.%m.%d %H:%M:%S")
    runtime_df["IMPORTANCE"] = aligned_metadata["news_max_importance"].fillna(0).astype(int)
    for col in scaled_feature_df.columns:
        runtime_df[col] = scaled_feature_df[col].astype(float).to_numpy()

    detail_df = pd.DataFrame()
    detail_df["BAR_TIME"] = aligned_metadata["bar_time"].dt.strftime("%Y.%m.%d %H:%M:%S")
    detail_df["EVENT_TIME"] = aligned_metadata["event_time"].dt.strftime("%Y.%m.%d %H:%M:%S").fillna("")
    detail_df["EVENT_NAME"] = _sanitize_csv_text(aligned_metadata["event_name"])
    detail_df["COUNTRY_CODE"] = _sanitize_csv_text(aligned_metadata["country_code"])
    detail_df["IMPORTANCE"] = aligned_metadata["importance"].fillna(0).astype(int)
    detail_df["MAX_IMPORTANCE"] = aligned_metadata["news_max_importance"].fillna(0).astype(int)
    detail_df["EVENT_COUNT"] = aligned_metadata["news_event_count"].fillna(0).astype(int)
    detail_df["OFFSET_HOURS"] = aligned_metadata["news_nearest_offset_hours"].fillna(0.0).astype(float)

    runtime_filename = config.runtime_events_path(symbol, timeframe).name
    detail_filename = config.runtime_event_details_path(symbol, timeframe).name

    _write_frame_to_targets(
        runtime_df,
        [
            config.runtime_events_path(symbol, timeframe, variant=variant),
            config.terminal_runtime_events_path(symbol, timeframe, variant=variant),
        ],
    )
    _write_frame_to_targets(
        detail_df,
        [
            config.runtime_event_details_path(symbol, timeframe, variant=variant),
            config.terminal_runtime_event_details_path(symbol, timeframe, variant=variant),
        ],
    )
