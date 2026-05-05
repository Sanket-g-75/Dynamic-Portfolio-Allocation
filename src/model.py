
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam
from src.pfutils import generate_weekly_allocations,  allocation_change_signals, build_daily_weights
from src.pfutils import compute_portfolio_returns



def build_model(lookback, num_features, num_stocks):
    inp = Input(shape=(lookback, num_features)) # Samples, features

    # CNN part: local temporal patterns in macro/FX/commodity/VIX features
    x = Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu')(inp) # Output = (lookback,num_features, 32)
    x = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(x)

    # LSTM part: longer-term temporal dependence
    x = LSTM(64, return_sequences=False)(x) # Output = (samples, 64)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Output: predicted future returns for each stock
    out = Dense(num_stocks, activation='linear', name='return_pred')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse',metrics=['accuracy','mse'])
    return model


# Forward Testing returning daily pf returns, cumulative value, allocations
def walk_forward_backtest(
    X_seq,
    y_seq,
    seq_dates,
    Y_close,
    tickers,
    train_window=1000,
    test_window=250,
    rebalance_weeks=1,
    temperature=0.1,
    max_turnover=None,
    transaction_cost_bps=0.0,
    epochs_per_step=5,
    batch_size=64
):
    """
    Returns:
        wf_port_ret:   Series of daily portfolio returns (net of costs)
        wf_port_curve: Series of cumulative value
        wf_allocations: dict with step -> allocations_df (rebalance-level)
    """
    num_samples = len(X_seq)
    start = train_window
    wf_port_ret_list = []
    wf_allocations = {}
    keras_model = None

    while start < num_samples - test_window:
        end_train = start
        end_test = min(start + test_window, num_samples)

        # Train slice
        X_tr = X_seq[end_train - train_window:end_train]
        y_tr = y_seq[end_train - train_window:end_train]

        # Corresponding dates for train/test slice
        dates_tr = seq_dates[end_train - train_window:end_train]
        dates_te = seq_dates[end_train:end_test]

        # Build & train model
        keras_model = build_model(
            lookback=X_seq.shape[1],
            num_features=X_seq.shape[2],
            num_stocks=y_seq.shape[1]
        )

        keras_model.fit(
            X_tr, y_tr,
            epochs=epochs_per_step,
            batch_size=batch_size,
            verbose=0
        )

        # Generate allocations on the test slice
        X_te = X_seq[end_train:end_test]
        alloc_df = generate_weekly_allocations(
            keras_model,
            X_te,
            dates_te,        # IMPORTANT: use dates of that slice
            tickers=tickers,
            rebalance_weeks=rebalance_weeks,
            temperature=temperature
        )

        wf_allocations[f"step_{start}"] = alloc_df

        # Signals
        signals_df = allocation_change_signals(
            alloc_df,
            change_threshold=0.1
        )

        # Daily weights + turnover
        daily_weights, daily_turnover = build_daily_weights(
            alloc_df,
            Y_close,
            signals_df=signals_df,
            max_turnover=max_turnover
        )

        # Restrict prices to the horizon we’re testing
        # (intersection of this test window’s dates with Y_close index)
        test_idx = pd.Index(dates_te)
        test_idx = test_idx.intersection(Y_close.index)

        dw_slice = daily_weights.loc[test_idx]
        dt_slice = daily_turnover.loc[test_idx]
        prices_slice = Y_close.loc[test_idx]

        # Portfolio returns for this step
        step_ret, step_curve = compute_portfolio_returns(
            dw_slice,
            prices_slice,
            daily_turnover=dt_slice,
            transaction_cost_bps=transaction_cost_bps
        )

        wf_port_ret_list.append(step_ret)

        # Move forward
        start += test_window

    # Stitch all steps
    wf_port_ret = pd.concat(wf_port_ret_list).sort_index()
    wf_port_curve = (1 + wf_port_ret).cumprod()

    return wf_port_ret, wf_port_curve, wf_allocations, keras_model
