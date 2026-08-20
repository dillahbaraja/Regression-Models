import config
from train_model import train_model_for_symbol
from evaluate_model import evaluate_model_for_symbol
from signal_generator import generate_signals_for_symbol
from hybrid_ensemble import build_hybrid_predictions
from backtest_helper import simple_python_backtest_for_symbol, simple_python_backtest_hybrid


def run_full_pipeline():
    config.ensure_directories()

    for symbol in config.training_symbols():
        train_model_for_symbol(symbol)

    for symbol in config.report_symbols():
        evaluate_model_for_symbol(symbol)

    for symbol in config.report_symbols():
        generate_signals_for_symbol(symbol)

    build_hybrid_predictions()

    for symbol in config.report_symbols():
        simple_python_backtest_for_symbol(symbol)

    simple_python_backtest_hybrid()


if __name__ == "__main__":
    run_full_pipeline()
