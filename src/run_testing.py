import config
from evaluate_model import evaluate_model_for_symbol
from signal_generator import generate_signals_for_symbol
from backtest_helper import simple_python_backtest_for_symbol


def run_testing_pipeline():
    config.ensure_directories()
    for symbol in config.report_symbols():
        evaluate_model_for_symbol(symbol)

    for symbol in config.report_symbols():
        generate_signals_for_symbol(symbol)

    for symbol in config.report_symbols():
        simple_python_backtest_for_symbol(symbol)


if __name__ == "__main__":
    run_testing_pipeline()
