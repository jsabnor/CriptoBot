import pandas as pd
import numpy as np

# Carga logs (ajusta paths si necesario)
trades = pd.read_csv('trades_log.csv')
equity = pd.read_csv('equity_curve.csv')

# Win Rate y Avg Profit (maneja unpaired)
buys = trades[trades['type'] == 'buy'].reset_index(drop=True)
sells = trades[trades['type'] == 'sell'].reset_index(drop=True)
num_pairs = min(len(buys), len(sells))
if num_pairs > 0:
    profits = (sells['price'][:num_pairs] * sells['qty'][:num_pairs] - buys['price'][:num_pairs] * buys['qty'][:num_pairs]) / (buys['price'][:num_pairs] * buys['qty'][:num_pairs]) * 100  # % return per trade
    win_rate = (profits > 0).mean() * 100
    avg_win = profits[profits > 0].mean()
    avg_loss = profits[profits <= 0].mean()
    print(f"Win Rate: {win_rate:.2f}% (basado en {num_pairs} pairs)")
    print(f"Avg Win: {avg_win:.2f}%")
    print(f"Avg Loss: {avg_loss:.2f}%")
else:
    print("No pairs completos para calcular win rate.")

# Max Drawdown
equity['equity'] = pd.to_numeric(equity['equity'], errors='coerce')
equity.dropna(inplace=True)
peak = equity['equity'].cummax()
drawdown = (equity['equity'] - peak) / peak * 100
max_dd = drawdown.min()
print(f"Max Drawdown: {max_dd:.2f}%")

# Sharpe Ratio (asumiendo daily returns, risk-free 0%, annualized)
equity['returns'] = equity['equity'].pct_change().fillna(0)
sharpe = equity['returns'].mean() / equity['returns'].std() * np.sqrt(252)  # Annualizado, ~252 trading days
print(f"Sharpe Ratio (annualizado): {sharpe:.2f}")