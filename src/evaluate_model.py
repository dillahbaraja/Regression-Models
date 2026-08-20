import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config

def evaluate_model_for_symbol(symbol):
    timeframe = config.resolve_pair_timeframe(symbol)
    print(f"Evaluating Model for {symbol} ({timeframe})...")
    
    X_test_scaled = pd.read_csv(config.processed_test_features_path(symbol, timeframe), index_col=0, parse_dates=True)
    test_df = pd.read_csv(config.processed_target_path(symbol, timeframe), index_col=0, parse_dates=True)
    y_test = test_df['target']
    
    results = []
    
    for model_type in config.MODEL_TYPES:
        try:
            model = joblib.load(config.model_path(symbol, timeframe, model_type))
        except FileNotFoundError:
            continue
            
        # Predict
        preds = model.predict(X_test_scaled)
        test_df[f'prediction_{model_type}'] = preds
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        # Directional Accuracy
        dir_acc = np.nan
        if config.TARGET_TYPE == "return":
            dir_acc = np.mean((y_test > 0) == (preds > 0)) * 100
            print(f"[{model_type}] Directional Accuracy: {dir_acc:.2f}%")
            
        print(f"[{model_type}] RMSE: {rmse:.5f}, MAE: {mae:.5f}, R2: {r2:.5f}")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            "symbol": [symbol],
            "timeframe": [timeframe],
            "model_type": [model_type],
            "RMSE": [rmse],
            "MAE": [mae],
            "R2": [r2],
            "Directional_Accuracy": [dir_acc]
        })
        metrics_df.to_csv(config.metrics_path(symbol, timeframe, model_type), index=False)
        
        # Save predictions for this model
        out_df = test_df[['target', f'prediction_{model_type}']].copy()
        out_df.rename(columns={f'prediction_{model_type}': 'prediction'}, inplace=True)
        out_df.to_csv(config.predictions_path(symbol, timeframe, model_type))
        
        results.append(metrics_df.iloc[0].to_dict())
        
    return results


def evaluate_all_models():
    config.ensure_directories()
    results = []
    for symbol in config.report_symbols():
        res = evaluate_model_for_symbol(symbol)
        results.extend(res)
    return results

if __name__ == "__main__":
    evaluate_all_models()
