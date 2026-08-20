# Backtest Result Interpretation

## Why the Current Backtest Result Looks Strong

The latest MT5 backtest result appears stronger than earlier training and testing attempts because the current system is more selective, more information-rich, and more aligned with the trading execution logic.

First, the Expert Advisor no longer evaluates every H1 candle. It trades only during macroeconomic news-event windows, specifically from `0H` to `+2H` after a relevant news release. This means the model is evaluated under market conditions where volatility and directional movement are more likely to appear. By avoiding ordinary non-event bars, the system reduces exposure to low-information market periods.

Second, the feature set is richer than the earlier technical-only versions. The model now uses OHLCV, technical indicators already present in the dataset, derived candle and volatility features, and macroeconomic-news features. The news features include event importance, actual value, forecast value, previous value, surprise terms, event counts, and separate aggregates for base-currency and quote-currency news. For forex forecasting, this gives the model information that is directly connected to currency repricing around economic releases.

Third, the prediction target is more realistic than the previous event-bar setup. With `ENTRY_DELAY_BARS = 1`, the model does not assume that the trade can be executed before or inside the same bar used to construct the features. The target return is measured from `close[t+1]` to `close[t+2]`, meaning the model learns the continuation or delayed reaction after the event bar has closed.

Fourth, the active window no longer uses `-1H`. The current setting is:

- `NEWS_WINDOW_BEFORE_HOURS = 0`
- `NEWS_WINDOW_AFTER_HOURS = 2`

This avoids using pre-event bars as tradable event samples. The model therefore focuses on event-time and post-event reactions, which is more consistent with the intended execution logic.

Fifth, the EA uses model-specific thresholds estimated from training predictions. It does not open trades for every small predicted return. A trade is opened only when the prediction exceeds the selected buy or sell threshold. This can reduce noisy entries and concentrate execution on higher-confidence forecast regions.

Sixth, the reported MT5 result is not primarily caused by an unusually high win rate. In the shown result, the win rate is below 50%, but the average winning trade is materially larger than the average losing trade. This means the profitability is driven by payoff asymmetry rather than by simply winning most trades. The system can remain profitable if profitable trades are large enough to compensate for more frequent smaller losses.

## Why This Does Not Indicate Training Leakage

Based on the checked artifacts, the model is trained only on data before `2024-01-01`, while the runtime event file used by the EA contains only testing-period rows from `2024-01-02` to `2026-06-30`. The checked `EURUSD_H1_runtime_events.csv` contained:

- total runtime rows: `6248`
- first bar: `2024-01-02 09:00:00`
- last bar: `2026-06-30 16:00:00`
- rows before 2024: `0`

The metadata also confirms:

- `train_end_date = 2024-01-01`
- `entry_delay_bars = 1`
- `prediction_horizon = 1`
- `news_window_before_hours = 0`
- `news_window_after_hours = 2`

Therefore, the current result does not appear to come from the earlier problem where training and testing periods overlapped. The testing period is treated as held-out historical data relative to the fitted model.

## How EUR/USD News Features Are Used During Training

For `EURUSD`, the model does not read news text directly. Instead, each relevant economic event is converted into numeric features that can be learned by the regression model.

Because the pair is `EURUSD`, the relevant news comes from:

- `EU` news for the EUR side
- `US` news for the USD side

For each H1 bar, the pipeline checks whether the bar falls inside the active event window, which is currently `0H` to `+2H` from the news release time. If the bar is inside that window, the model receives additional news-related values.

The news features can be understood as follows:

- `event count`: how many relevant EUR/USD news events occur around the bar
- `importance`: the highest importance level of the relevant news event
- `actual`: the released economic value
- `forecast`: the expected market value before release
- `previous`: the previous reported value
- `surprise actual-forecast`: the difference between the released value and the forecast
- `surprise actual-previous`: the difference between the released value and the previous value

For example, if a US CPI release has:

- actual = `3.2`
- forecast = `3.0`
- previous = `3.1`

Then the model receives:

- actual-forecast surprise = `0.2`
- actual-previous surprise = `0.1`

This is useful because financial markets often react not only to the released number itself, but also to whether the number is better or worse than expected.

During training, the model learns patterns such as:

If a high-importance USD event occurs, the actual value is higher than forecast, and the technical condition of `EURUSD` is in a certain state, what tends to happen to the next tradable return?

The target remains the delayed tradable return:

- features are taken from event bar `t`
- the trade is assumed after the bar closes
- the target return is measured from `close[t+1]` to `close[t+2]`

Thus, the model learns from historical examples in `2018-2023` how EUR/USD price movements reacted after relevant EUR and USD news events. In testing, the same type of feature vector is built for `2024-2026`, and the EA uses the trained model to predict whether the post-event return is strong enough to trigger a buy or sell decision.

## Remaining Cautions

The result should still be interpreted as a controlled historical backtest, not as proof of live profitability.

Several factors can still make historical performance look better than real trading:

- broker spread assumptions
- slippage during high-impact news
- execution delay
- commission and swap settings
- order rejection or requote behavior
- differences between historical news data availability and live news release timing
- sensitivity of results to stop-loss and take-profit settings

The next validation step should test whether the result remains robust under higher spread, added slippage, different news-importance filters, and pair-by-pair/model-by-model comparison.

## Short Interpretation for Manuscript Use

The strong backtest performance is likely driven by the event-driven sampling design, the integration of technical and macroeconomic features, delayed-entry target construction, and model-specific prediction thresholds. The result is more credible than earlier experiments because the testing runtime data contains only post-2024 held-out samples and the model no longer uses pre-event `-1H` samples. Nevertheless, the result should be presented as historical out-of-sample backtesting evidence and should be qualified with execution-cost and slippage sensitivity as future robustness tests.
