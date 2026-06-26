def SharpeLoss(weights,returns):
    portfolio_returns = (weights * returns).sum(dim=1)

    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    sharpe = mean_return / (volatility + 1e-8)

    return -sharpe