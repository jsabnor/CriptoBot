import pandas as pd

try:
    df = pd.read_csv('backtest_multi_results.csv')
    
    # Filtrar por timeframe 4h y los símbolos de producción
    target_symbols = ['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']
    target_tf = '4h'
    
    subset = df[(df['timeframe'] == target_tf) & (df['symbol'].isin(target_symbols))]
    
    print("=== RESULTADOS 4H (PRODUCCIÓN) ===")
    print(subset[['symbol', 'total_return', 'cagr', 'max_drawdown', 'sharpe_ratio', 'win_rate', 'profit_factor']].to_string(index=False))
    
    print("\n=== PROMEDIOS ===")
    print(subset[['total_return', 'cagr', 'max_drawdown', 'sharpe_ratio']].mean().to_string())
    
except Exception as e:
    print(f"Error: {e}")
