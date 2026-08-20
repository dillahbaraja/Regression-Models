import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import config


def _label_from_path(path):
    stem = Path(path).stem
    if stem.endswith("_tuning_trials"):
        stem = stem[: -len("_tuning_trials")]
    return stem


def plot_tuning_file(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Tuning log is empty: {csv_path}")

    label = _label_from_path(csv_path)
    figure_path = config.CHARTS_DIR / f"{label}_tuning_overview.png"

    score_col = "mean_validation_score"
    best_col = "best_score_so_far"
    fold_cols = sorted(
        [
            col
            for col in df.columns
            if col.startswith("fold_") and col.endswith("_score")
        ]
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].scatter(
        df["trial_number"],
        df[score_col],
        s=28,
        alpha=0.75,
        color="#1f4e79",
        label="Trial score",
    )
    axes[0].plot(
        df["trial_number"],
        df[best_col],
        color="#c00000",
        linewidth=2,
        label="Best score so far",
    )
    best_idx = df[score_col].idxmin()
    axes[0].scatter(
        [df.loc[best_idx, "trial_number"]],
        [df.loc[best_idx, score_col]],
        s=70,
        color="#2e8b57",
        zorder=3,
        label="Best trial",
    )
    axes[0].set_title(f"Tuning Search Overview: {label}")
    axes[0].set_xlabel("Trial / Candidate")
    axes[0].set_ylabel(config.OPTUNA_METRIC.upper())
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if fold_cols:
        for fold_col in fold_cols:
            fold_number = fold_col.split("_")[1]
            axes[1].plot(
                df["trial_number"],
                df[fold_col],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=f"Fold {fold_number}",
            )
        axes[1].plot(
            df["trial_number"],
            df[score_col],
            color="#111111",
            linewidth=2,
            linestyle="--",
            label="Mean validation score",
        )
        axes[1].set_title("Validation Score by Fold")
        axes[1].set_xlabel("Trial / Candidate")
        axes[1].set_ylabel(config.OPTUNA_METRIC.upper())
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=3, fontsize=8)
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "No fold-level score columns found.",
            ha="center",
            va="center",
            fontsize=12,
        )

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Tuning chart saved to {figure_path}")
    return figure_path


def iter_tuning_files(pattern=None):
    tuning_dir = config.TUNING_DIR
    if pattern:
        yield from sorted(tuning_dir.glob(pattern))
        return
    yield from sorted(tuning_dir.glob("*_tuning_trials.csv"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tuning overview charts from reports/tuning CSV logs."
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern inside reports/tuning, e.g. EURUSD_H1_xgboost*.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    files = list(iter_tuning_files(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No tuning trial CSV files found in {config.TUNING_DIR}."
        )
    for path in files:
        plot_tuning_file(path)
