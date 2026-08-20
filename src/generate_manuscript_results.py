import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


MANUSCRIPT_DIR = Path(r"C:\Users\dilla\OneDrive\Documents\Obsidian Vault\Regression Models")
FIGURES_DIR = MANUSCRIPT_DIR / "figures"
INITIAL_CAPITAL = 10000.0
MODEL_ORDER = ["xgboost", "lightgbm", "catboost", "sklearn_hgb", "voting"]
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "sklearn_hgb": "Sklearn HGB",
    "voting": "Ensemble Regression",
    "ensemble_regression": "Ensemble Regression",
}

TRANSACTION_COLUMNS = [
    "time",
    "deal_ticket",
    "order_ticket",
    "position_id",
    "symbol",
    "deal_type",
    "entry",
    "volume",
    "price",
    "sl",
    "tp",
    "profit",
    "swap",
    "commission",
    "comment",
]


def ensure_manuscript_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_text(value):
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace('"', "")
        .replace(",", ";")
        .strip()
    )


def load_transactions_robust(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip().split(",")
        expected = TRANSACTION_COLUMNS
        if [h.strip() for h in header] != expected:
            raise ValueError(f"Unexpected transaction header in {path}")

        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(",", 14)
            if len(parts) < 15:
                continue
            parts[14] = sanitize_text(parts[14])
            rows.append(parts[:15])

    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)


def load_predictive_metrics():
    rows = []
    for symbol in config.report_symbols():
        timeframe = config.resolve_pair_timeframe(symbol)
        for model_type in MODEL_ORDER:
            path = config.metrics_path(symbol, timeframe, model_type)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["model_label"] = MODEL_LABELS[model_type]
            rows.append(row)
    out = pd.DataFrame(rows)
    out["model_rank"] = out["model_type"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    out = out.sort_values(["symbol", "model_rank"]).drop(columns=["model_rank"])
    return out


def compute_max_drawdown_pct(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max
    return float(drawdown.max() * 100.0) if len(drawdown) else 0.0


def load_policy_metrics():
    rows = []
    curves = {}
    for tx_file in sorted((config.BACKTEST_DIR).glob("*_transactions.csv")):
        pair_key = tx_file.stem.replace("_transactions", "")
        symbol, timeframe, *model_parts = pair_key.split("_")
        model_type = "_".join(model_parts)
        if model_type not in MODEL_ORDER and model_type != "ensemble_regression":
            continue

        tx = load_transactions_robust(tx_file)
        exit_tx = tx[tx["entry"].astype(str).str.upper().eq("OUT")].copy()
        if exit_tx.empty:
            continue

        exit_tx["time"] = pd.to_datetime(exit_tx["time"])
        exit_tx["net_profit"] = (
            pd.to_numeric(exit_tx["profit"], errors="coerce").fillna(0.0)
            + pd.to_numeric(exit_tx["swap"], errors="coerce").fillna(0.0)
            + pd.to_numeric(exit_tx["commission"], errors="coerce").fillna(0.0)
        )
        returns = exit_tx["net_profit"] / INITIAL_CAPITAL
        equity = INITIAL_CAPITAL + exit_tx["net_profit"].cumsum()
        curves[(symbol, model_type)] = pd.DataFrame({"time": exit_tx["time"], "equity": equity})

        gross_profit = float(exit_tx.loc[exit_tx["net_profit"] > 0, "net_profit"].sum())
        gross_loss = float(exit_tx.loc[exit_tx["net_profit"] < 0, "net_profit"].sum())
        win_rate = float((exit_tx["net_profit"] > 0).mean() * 100.0)
        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else math.inf
        avg_win = float(exit_tx.loc[exit_tx["net_profit"] > 0, "net_profit"].mean())
        avg_loss = float(exit_tx.loc[exit_tx["net_profit"] < 0, "net_profit"].mean())
        risk_reward = float(avg_win / abs(avg_loss)) if avg_loss < 0 else math.inf
        sharpe = float((returns.mean() / returns.std(ddof=1)) * math.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0

        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": model_type,
                "model_label": MODEL_LABELS.get(model_type, model_type),
                "closed_trades": int(len(exit_tx)),
                "net_profit": float(exit_tx["net_profit"].sum()),
                "gross_profit": gross_profit,
                "gross_loss": abs(gross_loss),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "return_pct": float(exit_tx["net_profit"].sum() / INITIAL_CAPITAL * 100.0),
                "expectancy": float(exit_tx["net_profit"].mean()),
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": compute_max_drawdown_pct(equity.to_numpy()),
                "risk_reward_ratio": risk_reward,
            }
        )

    out = pd.DataFrame(rows)
    out["model_rank"] = out["model_type"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
    out = out.sort_values(["symbol", "model_rank"]).drop(columns=["model_rank"])
    return out, curves


def save_summary_tables(predictive_df, policy_df):
    predictive_df.to_csv(MANUSCRIPT_DIR / "predictive_metrics_summary.csv", index=False)
    policy_df.to_csv(MANUSCRIPT_DIR / "backtest_metrics_summary.csv", index=False)


def render_predictive_overview(predictive_df):
    pairs = list(config.report_symbols())
    models = MODEL_ORDER

    rmse = predictive_df.pivot(index="symbol", columns="model_type", values="RMSE").reindex(index=pairs, columns=models)
    da = predictive_df.pivot(index="symbol", columns="model_type", values="Directional_Accuracy").reindex(index=pairs, columns=models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    for ax, data, title, fmt in [
        (axes[0], rmse, "RMSE", "{:.5f}"),
        (axes[1], da, "Directional Accuracy (%)", "{:.2f}"),
    ]:
        im = ax.imshow(data.to_numpy(), aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=30, ha="right")
        ax.set_yticks(range(len(pairs)), pairs)
        ax.set_title(title, fontsize=12, weight="bold")
        for i in range(len(pairs)):
            for j in range(len(models)):
                value = data.iloc[i, j]
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Held-Out Predictive Performance Across Five Currency Pairs", fontsize=15, weight="bold")
    fig.savefig(FIGURES_DIR / "training_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_actual_predicted_return_frame(symbol: str, model_type: str) -> pd.DataFrame:
    timeframe = config.resolve_pair_timeframe(symbol)
    prediction_path = config.predictions_path(symbol, timeframe, model_type)
    pred_df = pd.read_csv(prediction_path, parse_dates=["time"]).set_index("time")
    frame = pd.DataFrame(index=pred_df.index)
    frame["actual_return"] = pred_df["target"].astype(float)
    frame["predicted_return"] = pred_df["prediction"].astype(float)
    frame.dropna(inplace=True)
    return frame


def render_actual_vs_predicted_return(predictive_df):
    best_rows = (
        predictive_df.sort_values(["symbol", "RMSE"])
        .groupby("symbol", as_index=False)
        .first()
    )

    fig, axes = plt.subplots(len(best_rows), 1, figsize=(12, 16), constrained_layout=True)
    if len(best_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, best_rows.itertuples(index=False)):
        frame = build_actual_predicted_return_frame(row.symbol, row.model_type)
        actual = frame["actual_return"].to_numpy()
        predicted = frame["predicted_return"].to_numpy()
        limit = max(np.abs(actual).max(), np.abs(predicted).max())
        limit = max(limit, 1e-6)

        ax.scatter(actual, predicted, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")
        ax.plot([-limit, limit], [-limit, limit], linestyle="--", linewidth=1.2, color="#d62728", label="Ideal fit")
        ax.set_title(f"{row.symbol} | {row.model_label}", fontsize=11, weight="bold")
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", frameon=True)

        per_pair_path = FIGURES_DIR / f"actual_vs_predicted_return_{row.symbol}.png"
        pair_fig, pair_ax = plt.subplots(figsize=(11, 4.6))
        pair_ax.scatter(actual, predicted, s=11, alpha=0.35, color="#1f77b4", edgecolors="none")
        pair_ax.plot([-limit, limit], [-limit, limit], linestyle="--", linewidth=1.3, color="#d62728", label="Ideal fit")
        pair_ax.set_title(
            f"Actual vs Predicted Return on Held-Out Event Windows - {row.symbol} ({row.model_label})",
            fontsize=13,
            weight="bold",
        )
        pair_ax.set_xlabel("Actual Return")
        pair_ax.set_ylabel("Predicted Return")
        pair_ax.set_xlim(-limit, limit)
        pair_ax.set_ylim(-limit, limit)
        pair_ax.grid(True, alpha=0.25)
        pair_ax.legend(frameon=True)
        pair_fig.savefig(per_pair_path, dpi=220, bbox_inches="tight")
        plt.close(pair_fig)

    fig.suptitle("Actual vs Predicted Return for the Best-RMSE Model in Each Pair", fontsize=15, weight="bold")
    fig.savefig(FIGURES_DIR / "actual_vs_predicted_return_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_cumulative_profit_curves(curves):
    for symbol in config.report_symbols():
        fig, ax = plt.subplots(figsize=(11, 5.2))
        plotted = False
        for model_type in MODEL_ORDER:
            curve = curves.get((symbol, model_type))
            if curve is None or curve.empty:
                continue
            plotted = True
            ax.plot(curve["time"], curve["equity"], linewidth=1.6, label=MODEL_LABELS[model_type])
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Cumulative Equity Curve by Model Families - {symbol}", fontsize=14, weight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity (Initial Capital = 10,000)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, frameon=True)
        fig.autofmt_xdate()
        fig.savefig(FIGURES_DIR / f"cumulative_profit_{symbol}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def render_data_coverage_split():
    symbols = list(config.report_symbols())
    ranges = []
    for idx, symbol in enumerate(symbols):
        dataset = pd.read_csv(config.dataset_data_path(symbol), nrows=5)
        time_col = next(col for col in dataset.columns if col.lower() == "time")
        time_index = pd.to_datetime(pd.read_csv(config.dataset_data_path(symbol), usecols=[time_col])[time_col], errors="coerce")
        ranges.append((symbol, time_index.min(), time_index.max(), idx))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for symbol, start, end, idx in ranges:
        ax.plot([start, end], [idx, idx], linewidth=10, solid_capstyle="butt", color="#9ecae1")
        ax.plot([start, pd.Timestamp(config.TRAIN_END_DATE) - pd.Timedelta(hours=1)], [idx, idx], linewidth=10, solid_capstyle="butt", color="#2ca25f")
        ax.plot([pd.Timestamp(config.TRAIN_END_DATE), end], [idx, idx], linewidth=10, solid_capstyle="butt", color="#de2d26")

    split_ts = pd.Timestamp(config.TRAIN_END_DATE)
    ax.axvline(split_ts, color="black", linestyle="--", linewidth=1.5, label="Train/Test Split")
    ax.set_yticks(range(len(symbols)), symbols)
    ax.set_title("Historical Coverage and Chronological Split Across Five Currency Pairs", fontsize=14, weight="bold")
    ax.set_xlabel("Date")
    ax.grid(True, axis="x", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color="#2ca25f", linewidth=8, label="Training Window"),
        plt.Line2D([0], [0], color="#de2d26", linewidth=8, label="Testing Window"),
        plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Split at 2024-01-01"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    fig.autofmt_xdate()
    fig.savefig(FIGURES_DIR / "data_coverage_split.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_box_workflow(path, title, boxes, arrows):
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=220)
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", pad=15)
    for box in boxes:
        x, y, w, h, text_val, color = box
        rect = patches.FancyBboxPatch(
            (x + 0.005, y + 0.005), w - 0.01, h - 0.01,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="#555555", linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text_val, ha="center", va="center", fontsize=9.5, wrap=True)
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(arrowstyle="-|>", linewidth=1.5, color="#555555", mutation_scale=12))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_methodology_figures():
    render_data_coverage_split()

    render_box_workflow(
        FIGURES_DIR / "feature_engineering_pipeline.png",
        "Feature Engineering and Event-Window Construction",
        [
            (0.02, 0.45, 0.15, 0.24, "OHLCV H1 Bars", "#c6dbef"),
            (0.21, 0.45, 0.18, 0.24, "Embedded Technical Indicators", "#9ecae1"),
            (0.43, 0.45, 0.18, 0.24, "Derived Candle and Return Features", "#6baed6"),
            (0.65, 0.45, 0.15, 0.24, "Macroeconomic News Features", "#74c476"),
            (0.84, 0.45, 0.14, 0.24, "Event-Window Feature Matrix", "#31a354"),
        ],
        [
            (0.17, 0.57, 0.21, 0.57),
            (0.39, 0.57, 0.43, 0.57),
            (0.61, 0.57, 0.65, 0.57),
            (0.80, 0.57, 0.84, 0.57),
        ],
    )

    render_box_workflow(
        FIGURES_DIR / "research_workflow.png",
        "Research Workflow for the Event-Driven Five-Pair Study",
        [
            (0.02, 0.46, 0.12, 0.22, "Pair Datasets", "#c6dbef"),
            (0.17, 0.46, 0.12, 0.22, "News Dataset", "#c7e9c0"),
            (0.32, 0.46, 0.15, 0.22, "Merged Event-Driven Features", "#9ecae1"),
            (0.50, 0.46, 0.16, 0.22, "Chronological Split and Optuna Tuning", "#6baed6"),
            (0.69, 0.46, 0.12, 0.22, "Model Training", "#4292c6"),
            (0.84, 0.46, 0.14, 0.22, "MT5 and Statistical Evaluation", "#238b45"),
        ],
        [
            (0.14, 0.57, 0.32, 0.57),
            (0.29, 0.57, 0.32, 0.57),
            (0.47, 0.57, 0.50, 0.57),
            (0.66, 0.57, 0.69, 0.57),
            (0.81, 0.57, 0.84, 0.57),
        ],
    )

def main():
    ensure_manuscript_dirs()
    predictive_df = load_predictive_metrics()
    policy_df, curves = load_policy_metrics()
    save_summary_tables(predictive_df, policy_df)
    render_predictive_overview(predictive_df)
    render_cumulative_profit_curves(curves)
    render_actual_vs_predicted_return(predictive_df)
    render_methodology_figures()
    print(f"Predictive rows: {len(predictive_df)}")
    print(f"Policy rows: {len(policy_df)}")


if __name__ == "__main__":
    main()
