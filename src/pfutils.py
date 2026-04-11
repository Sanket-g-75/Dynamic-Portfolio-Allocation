import pandas as pd
import numpy as np


# Function to return weights
def returns_to_weights(pred_returns, temperature=0.1):
    scaled = pred_returns / temperature
    exp_r = np.exp(scaled - np.max(scaled))  # numerical stability
    w = exp_r / exp_r.sum()
    return w


# Function to generate weekly allocations per stocks
def generate_weekly_allocations(
    model, X_seq, seq_dates,
    tickers, rebalance_weeks=1,temperature=0.1):

    step = rebalance_weeks * 5
    reb_dates = []
    reb_weights = []

    i = 0
    while i < len(X_seq):
        x = X_seq[i][np.newaxis, ...]  # (1, lookback, features)
        pred = model.predict(x, verbose=0)[0]  # (num_stocks,)
        w = returns_to_weights(pred, temperature=temperature)
        reb_dates.append(seq_dates[i])
        reb_weights.append(w)
        i += step

    alloc_df = pd.DataFrame(reb_weights, index=reb_dates, columns=tickers)
    return alloc_df


# Function to compute allocation change signals
def allocation_change_signals(alloc_df, change_threshold=0.05):

    signals = []
    prev_w = None

    for date, w in alloc_df.iterrows():
        w_vec = w.values
        if prev_w is None:
            signals.append(1)  # first rebalance -> allocate
        else:
            l1_change = np.abs(w_vec - prev_w).sum()
            signals.append(1 if l1_change > change_threshold else 0)
        prev_w = w_vec

    signal_df = alloc_df.copy()
    signal_df['signal'] = signals
    return signal_df


# Computes Daily, Cumulative Portfolio Returns
def compute_portfolio_returns(
    daily_weights,
    Y_close,
    daily_turnover=None,
    transaction_cost_bps=0.0
):
    """
    daily_weights > DataFrame [dates x tickers], weights per day
    Y_close > DataFrame [dates x tickers], close prices
    daily_turnover > Series [dates], from build_daily_weights; if None -> no cost
    transaction_cost_bps > float, cost per 1-way trade in basis points (bps).
                          Example: 10 bps = 0.10% per notional traded.

    Returns:
        port_daily_ret: Series of daily net returns (after costs)
        port_curve:     Series of cumulative portfolio value (starting at 1.0)
    """
    tickers = [c for c in daily_weights.columns if c in Y_close.columns]
    daily_ret = Y_close[tickers].pct_change().fillna(0)

    # Align
    daily_weights = daily_weights.reindex(daily_ret.index).fillna(method='ffill')

    # Gross portfolio returns
    gross_ret = (daily_weights * daily_ret).sum(axis=1)

    if daily_turnover is None:
        net_ret = gross_ret
    else:
        daily_turnover = daily_turnover.reindex(daily_ret.index).fillna(0)
        # cost per day = turnover * cost_rate
        cost_rate = transaction_cost_bps / 10000.0
        cost = daily_turnover * cost_rate
        net_ret = gross_ret - cost

    port_curve = (1 + net_ret).cumprod()

    return net_ret, port_curve


# Builds Daily Weights for the stocks for allocation
def build_daily_weights(
    allocations_df,
    Y_close,
    signals_df=None,
    max_turnover=None
):
    """
    allocations_df: rebalance weights, index = rebalance dates, cols = tickers
    Y_close:        daily close prices, index = all trading dates, cols = tickers
    signals_df:     optional, same index as allocations_df with 'signal' col
    max_turnover:   optional, max allowed L1 change in weights per day (e.g. 0.5)
                    If None -> no cap.

    Returns:
        daily_weights: DataFrame [dates x tickers]
        daily_turnover: Series [dates] (L1 turnover / 2)
    """
    tickers = [c for c in allocations_df.columns if c in Y_close.columns]

    # Daily returns index
    daily_idx = Y_close.index

    # 1) Start from rebalance weights, ffill to daily
    raw_daily = allocations_df[tickers].reindex(daily_idx, method='ffill')

    # 2) Handle signals: if signal=0, keep previous weights (no new rebalance)
    if signals_df is not None and 'signal' in signals_df.columns:
        sig_daily = signals_df['signal'].reindex(daily_idx, method='ffill').fillna(0)

        adj = raw_daily.copy()
        for i in range(1, len(adj)):
            if sig_daily.iloc[i] == 0:
                # keep yesterday's weights
                adj.iloc[i] = adj.iloc[i-1]
        raw_daily = adj

    # 3) Apply turnover cap (if any)
    daily_weights = raw_daily.copy()
    daily_turnover = pd.Series(index=daily_idx, dtype=float)

    prev_w = None
    for i, (dt, w) in enumerate(daily_weights.iterrows()):
        w_vec = w.values.astype(float)

        if prev_w is None:
            # normalize first day, zero turnover
            if w_vec.sum() != 0:
                w_vec = w_vec / w_vec.sum()
            daily_weights.iloc[i] = w_vec
            daily_turnover.iloc[i] = 0.0
            prev_w = w_vec
            continue

        # target change
        delta = w_vec - prev_w
        l1_change = np.abs(delta).sum()

        # cap turnover if requested
        if (max_turnover is not None) and (l1_change > max_turnover) and (l1_change > 0):
            scale = max_turnover / l1_change
            w_vec = prev_w + scale * delta
            # re-normalize to sum 1
            if w_vec.sum() != 0:
                w_vec = w_vec / w_vec.sum()
            delta = w_vec - prev_w
            l1_change = np.abs(delta).sum()

        # store
        daily_weights.iloc[i] = w_vec
        # standard convention: turnover = 0.5 * L1 (buy+sell)
        daily_turnover.iloc[i] = 0.5 * l1_change
        prev_w = w_vec

    return daily_weights, daily_turnover

def compute_drawdown(port_curve):

    running_max = port_curve.cummax()
    drawdown = port_curve / running_max - 1.0
    max_dd = drawdown.min()
    return drawdown, max_dd

# Computes Computation Stats (Volatility, Returns, Sharpe, etc)
def compute_stats(
    port_daily_ret,
    risk_free_rate=0.0,
    periods_per_year=252
):

    # Basic stats
    avg_ret = port_daily_ret.mean()
    vol = port_daily_ret.std()

    # Annualization
    ann_ret = (1 + avg_ret) ** periods_per_year - 1
    ann_vol = vol * np.sqrt(periods_per_year)

    # Sharpe
    rf_daily = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    excess_daily = port_daily_ret - rf_daily
    excess_ann_ret = (1 + excess_daily.mean()) ** periods_per_year - 1
    sharpe = excess_ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Sortino
    downside = port_daily_ret[port_daily_ret < 0]
    downside_vol = downside.std()
    sortino = excess_ann_ret / (downside_vol * np.sqrt(periods_per_year)) if downside_vol > 0 else np.nan

    return {
        "avg_daily_ret": avg_ret,
        "daily_vol": vol,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino
    }


# Built a function to build Backtest Report
def backtest_report(
    port_daily_ret,
    port_curve,
    risk_free_rate=0.0,
    periods_per_year=252
):
    drawdown, max_dd = compute_drawdown(port_curve)
    stats = compute_stats(port_daily_ret, risk_free_rate, periods_per_year)

    cagr = stats["ann_ret"]
    ann_vol = stats["ann_vol"]
    sharpe = stats["sharpe"]
    sortino = stats["sortino"]

    calmar = -cagr / max_dd if max_dd < 0 else np.nan

    report = {
        "CAGR": cagr,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Final Value": port_curve.iloc[-1],
        "Start Date": port_curve.index[0],
        "End Date": port_curve.index[-1]
    }

    return report, drawdown


# Compare model performance with equal weight stocks
def compare_with_equal_weight(
    model_port_ret,
    model_port_curve,
    Y_close,
    tickers,
    risk_free_rate=0.0
):

    # --- Equal-Weight Portfolio (EW) ---
    ew_w = np.array([1/len(tickers)] * len(tickers))
    daily_ret = Y_close[tickers].pct_change().fillna(0)

    ew_ret = (daily_ret * ew_w).sum(axis=1)
    ew_curve = (1 + ew_ret).cumprod()

    # --- Compute stats ---
    model_report, model_dd = backtest_report(model_port_ret, model_port_curve, risk_free_rate)
    ew_report, ew_dd = backtest_report(ew_ret, ew_curve, risk_free_rate)

    comparison = pd.DataFrame({
        "Model": model_report,
        "EqualWeight": ew_report
    })

    return comparison, ew_ret, ew_curve
