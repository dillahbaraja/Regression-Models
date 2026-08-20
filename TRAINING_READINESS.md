# Training Readiness Notes

## Training Scope

The current published training snapshot is aligned with the following four currency pairs:

- `AUDUSD`
- `EURUSD`
- `GBPUSD`
- `USDJPY`

Detected dataset files:

- `dataset/AUDUSD_H1_Data.csv`
- `dataset/EURUSD_H1_Data.csv`
- `dataset/GBPUSD_H1_Data.csv`
- `dataset/USDJPY_H1_Data.csv`
- `dataset/NEWS_2018.01.01-2026.06.30_ALL_.csv`

## Historical Data Coverage

The raw historical range loaded by the pipeline spans:

- Start: `2018-01-01`
- End: `2026-06-30`

Observed H1 bar coverage in the current pair datasets:

- `AUDUSD`: `2018-01-02 00:00:00` to `2026-06-30 23:00:00`
- `EURUSD`: `2018-01-02 00:00:00` to `2026-06-30 23:00:00`
- `GBPUSD`: `2018-01-02 00:00:00` to `2026-06-30 23:00:00`
- `USDJPY`: `2018-01-02 00:00:00` to `2026-06-30 23:00:00`

## Train-Test Split

The split is date-based and leakage-safe:

- Training period: `2018-01-01` to `2023-12-31`
- Testing period: `2024-01-01` to `2026-06-30`

Implemented rule in code:

- rows with timestamp `< 2024-01-01` go to training
- rows with timestamp `>= 2024-01-01` go to testing

## Event-Window Rule

The training and testing samples are not all H1 bars. Only bars inside the configured economic-news window are used.

Current event window:

- `0H` to `+2H` from the release time

This means the present setup does **not** use `-1H` anymore.

## Target Definition

The task remains regression, with:

- `TARGET_TYPE = return`
- `PREDICTION_HORIZON = 1`
- `ENTRY_DELAY_BARS = 1`

Operational meaning:

- features are taken from event bar `t`
- trading is assumed only after bar `t` has closed
- the tradable target return is measured from close `t+1` to close `t+2`

This is more realistic than predicting a return that starts from the same bar used to construct the features.

## Features Used

The pipeline uses three feature groups.

### 1. Price and volume inputs

- `open`
- `high`
- `low`
- `close`
- `tick_volume`

### 2. Technical indicators already present in the dataset

Examples verified from the current files:

- `rsi`
- `atr`
- `macd_main`
- `macd_signal`
- `bands_upper`
- `bands_lower`
- `stoch_main`
- `stoch_signal`
- `adx_main`
- `adx_plus_di`
- `adx_minus_di`
- `rsi_h4`
- `ma_h4`
- `rsi_m15`
- `atr_m15`
- `hour`
- `day_of_week`

### 3. Derived technical and candle features

- `body`
- `upper_wick`
- `lower_wick`
- `log_return`
- `volatility_20`

### 4. News and fundamental features

The news data is loaded from:

- `dataset/NEWS_2018.01.01-2026.06.30_ALL_.csv`

For each pair, relevant news is selected based on the two currencies in the symbol. The resulting features include:

- `news_event_present`
- `news_event_count`
- `news_max_importance`
- `news_nearest_offset_hours`
- `base_news_count`
- `base_max_importance`
- `base_actual_sum`
- `base_forecast_sum`
- `base_previous_sum`
- `base_surprise_actual_forecast_sum`
- `base_surprise_actual_previous_sum`
- `quote_news_count`
- `quote_max_importance`
- `quote_actual_sum`
- `quote_forecast_sum`
- `quote_previous_sum`
- `quote_surprise_actual_forecast_sum`
- `quote_surprise_actual_previous_sum`

## Models Trained

The configured model families are:

- `xgboost`
- `lightgbm`
- `catboost`
- `sklearn_hgb`
- `voting`

The `voting` model is used as the ensemble regression benchmark.

## Hyperparameter Tuning

Hyperparameter optimization uses:

- Optuna
- `50` trials per model

## Verified Event Sample Counts

The current datasets produce the following event-window sample counts after feature building and target generation:

- `AUDUSD`: total event rows `18901`, train `13718`, test `5183`
- `EURUSD`: total event rows `21819`, train `15571`, test `6248`
- `GBPUSD`: total event rows `20031`, train `14215`, test `5816`
- `USDJPY`: total event rows `21338`, train `15249`, test `6089`

These counts confirm that every published pair currently has non-empty training and testing samples under the configured news-event window.

## Output Locations

Primary project output folder:

- `Assets`

MT5 runtime mirror folder:

- `C:\Users\dilla\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Regression_Assets`

The project uses `Assets` as the main source of generated artifacts, while the MT5 Common `Regression_Assets` location is kept as the runtime folder so the EA can read the files correctly during testing and execution.

## Why This Configuration Is Used

This configuration is used for four main reasons.

First, the training and testing sets are separated chronologically so the testing period is treated as held-out future data relative to the training period. This prevents the earlier mistake of mixing or extending training into the evaluation phase.

Second, the event-window restriction keeps the learning problem aligned with the actual trading design. The EA is intended to trade around macroeconomic events, so the model should be trained on those conditions rather than on all ordinary H1 bars.

Third, `ENTRY_DELAY_BARS = 1` makes the regression target more realistic. It avoids building a target from price movement that would begin before the decision could actually be executed.

Fourth, combining OHLCV, technical indicators already exported in the dataset, and macroeconomic-news features keeps the pipeline consistent with the intended hybrid technical-fundamental forecasting setup.

## Current Status

The pipeline is ready to run, based on the following checks:

- dataset files are present
- training symbols are detected correctly
- all detected pairs load successfully
- required feature columns are present
- news data file is present
- each pair has non-empty train and test event samples
- output directories are configured
- training batch file is prepared

## Command to Run

Run:

- `start_training.bat`
