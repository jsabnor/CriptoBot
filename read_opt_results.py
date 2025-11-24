import pandas as pd

try:
    df = pd.read_csv('optimization_results.csv')
    print("=== TOP 10 CONFIGURATIONS (BY SCORE) ===")
    print(df.head(10).to_string(index=False))
    
    print("\n=== TOP 10 CONFIGURATIONS (BY ROI) ===")
    print(df.sort_values('avg_roi', ascending=False).head(10).to_string(index=False))
except Exception as e:
    print(f"Error reading results: {e}")
