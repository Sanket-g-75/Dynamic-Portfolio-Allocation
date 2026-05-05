import os
import pandas as pd
import numpy as np
import tensorflow as tf

import mlflow
import mlflow.keras

from src.model import walk_forward_backtest
from src.pfutils import compare_with_equal_weight
from src.preprocess import scaling

def create_sequences(X, y, dates, lookback):
    """
    Generate sequences of shape (samples, lookback, features) for the CNN-LSTM.
    """
    X_seq, y_seq, seq_dates = [], [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
        seq_dates.append(dates[i+lookback])
    return np.array(X_seq), np.array(y_seq), np.array(seq_dates)

mlflow.set_experiment("e2e-mlops-project")

def main():
    print("Loading dataset...")
    # Using relative path based on the root directory
    data_path = os.path.join('data', 'final', 'Dataset.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Forward fill to handle any missing prices, then drop rows that are still NaN
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    # Identify tradeable tickers (closing prices). Exclude VIX from the tradeable universe.
    tickers = [c for c in df.columns if c.endswith('-Close') and 'VIX' not in c]
    Y_close = df[tickers]
    
    # Calculate future returns (e.g., 5-day forward return) as the target for the model
    print("Preparing targets and features...")
    y = Y_close.pct_change(5).shift(-5)
    
    # Scale features using the utility from preprocess
    X_data = scaling(df.values)
    X_df = pd.DataFrame(X_data, index=df.index, columns=df.columns)
    
    # Align X and y by dropping the last few rows with NaN targets
    valid_idx = y.dropna().index
    X_df = X_df.loc[valid_idx]
    y = y.loc[valid_idx]
    
    # Create 3D sequences for the model (lookback window of 60 days)
    lookback = 60
    X_seq, y_seq, seq_dates = create_sequences(X_df.values, y.values, valid_idx, lookback)
    
    print(f"Data prepared: X_seq shape {X_seq.shape}, y_seq shape {y_seq.shape}")
    
    # Hyperparameters
    params = {
        'lookback': lookback,
        'train_window': 1000,
        'test_window': 250,
        'rebalance_weeks': 1,
        'temperature': 0.1,
        'max_turnover': 0.5,
        'transaction_cost_bps': 5.0,
        'epochs_per_step': 10,
        'batch_size': 64
    }
    
    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_artifact(data_path, artifact_path="data")
        run_id = run.info.run_id
        print("Run ID:", run_id)
        # Train the model using the walk-forward backtest pipeline from model.py
        # This evaluates the model fairly over time
        print("Starting walk-forward backtest training... This may take a few minutes.")
        wf_port_ret, wf_port_curve, wf_allocations, keras_model = walk_forward_backtest(
            X_seq=X_seq,
            y_seq=y_seq,
            seq_dates=seq_dates,
            Y_close=Y_close,
            tickers=tickers,
            train_window=params['train_window'],
            test_window=params['test_window'],
            rebalance_weeks=params['rebalance_weeks'],
            temperature=params['temperature'],
            max_turnover=params['max_turnover'],
            transaction_cost_bps=params['transaction_cost_bps'],
            epochs_per_step=params['epochs_per_step'],
            batch_size=params['batch_size']
        )
        
        # Generate backtest report and compare with equal weight benchmark
        print("\nGenerating backtest report...")
        comparison, ew_ret, ew_curve = compare_with_equal_weight(
            model_port_ret=wf_port_ret,
            model_port_curve=wf_port_curve,
            Y_close=Y_close,
            tickers=tickers
        )
        
        print("\n--- Strategy vs Equal Weight Comparison ---")
        print(comparison)
        
        # Log metrics to MLflow
        # 'comparison' is a DataFrame with 'Model' and 'EqualWeight' columns. 
        # We extract the 'Model' stats for logging.
        metrics = comparison['Model'].to_dict()
        # Some metrics might be dates (Start Date, End Date), so we filter them out
        metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, (str, pd.Timestamp))}
        mlflow.log_metrics(metrics)
        
        # Save the allocation and performance curves
        print("\nSaving results...")
        os.makedirs('results', exist_ok=True)
        wf_port_curve.to_csv(os.path.join('results', 'model_portfolio_curve.csv'))
        ew_curve.to_csv(os.path.join('results', 'equal_weight_curve.csv'))
        
        # Save step-by-step allocations
        allocations_list = []
        for step, alloc in wf_allocations.items():
            alloc['step'] = step
            allocations_list.append(alloc)
        if allocations_list:
            combined_alloc = pd.concat(allocations_list)
            combined_alloc.to_csv(os.path.join('results', 'allocations.csv'))

        # Log results directory to MLflow
        mlflow.log_artifacts('results', artifact_path='results')
        
        # Log the final Keras model
        if keras_model is not None:
            mlflow.keras.log_model(keras_model, "final_model")

        print("Training and backtest completed successfully.")

if __name__ == '__main__':
    main()
