import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import config


MODEL_ORDER = [
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
    ("catboost", "CatBoost"),
    ("sklearn_hgb", "Sklearn HGB"),
]


def chart_path(symbol, timeframe, model_type):
    name = f"{config.pair_key(symbol, timeframe)}_{model_type}_tuning_overview.png"
    return config.CHARTS_DIR / name


def combine_pair(symbol, timeframe=None):
    timeframe = timeframe or config.resolve_pair_timeframe(symbol)
    files = []
    for model_type, label in MODEL_ORDER:
        path = chart_path(symbol, timeframe, model_type)
        if not path.exists():
            raise FileNotFoundError(f"Missing chart for {symbol} / {model_type}: {path}")
        files.append((path, label))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    for ax, (path, label) in zip(axes, files):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(label, fontsize=13, pad=10)
        ax.axis("off")

    fig.suptitle(
        f"Hyperparameter Tuning Overview for {symbol} ({timeframe})",
        fontsize=18,
        fontweight="bold",
    )

    output_path = config.CHARTS_DIR / f"{config.pair_key(symbol, timeframe)}_combined_tuning_overview.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined pair chart saved to {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine individual tuning charts into one 2x2 figure per pair."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(config.REPORT_SYMBOLS),
        help="Symbols to combine. Default: report symbols only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for symbol in args.symbols:
        combine_pair(symbol)
