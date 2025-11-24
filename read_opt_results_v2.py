import pandas as pd

try:
    # Resultados promedio
    df = pd.read_csv('optimization_results.csv')
    print("=== TOP 10 CONFIGURATIONS (BY SCORE) ===")
    print(df.head(10).to_string(index=False))
    
    print("\n=== TOP 10 CONFIGURATIONS (BY ROI) ===")
    print(df.sort_values('avg_roi', ascending=False).head(10).to_string(index=False))
    
    # Ranking de los mejores 3
    print("\n🏆 TOP 3 CONFIGURATIONS (BY SCORE):")
    print(df.head(3).to_string(index=False))
    
    print("\n💰 TOP 3 CONFIGURATIONS (BY ROI):")
    print(df.sort_values('avg_roi', ascending=False).head(3).to_string(index=False))
    
    # Resultados por par
    print("\n=== RESULTS BY PAIR ===")
    df_pairs = pd.read_csv('optimization_results_by_pair.csv')
    for symbol in df_pairs['symbol'].unique():
        print(f"\n🔹 {symbol} - TOP 3 BY ROI")
        print(df_pairs[df_pairs['symbol'] == symbol].sort_values('roi', ascending=False).head(3).to_string(index=False))
        
        print(f"\n🔹 {symbol} - TOP 3 BY SCORE")
        print(df_pairs[df_pairs['symbol'] == symbol].sort_values('score', ascending=False).head(3).to_string(index=False))

except Exception as e:
    print(f"Error reading results: {e}")
