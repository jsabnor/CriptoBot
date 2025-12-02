import pandas as pd
from data_cache import DataCache

# Verificar datos en cache
cache = DataCache()
df = cache.get_data('ETH/USDT', '1h')

print(f"\n{'='*60}")
print("INFORMACIÓN DEL CACHE ETH/USDT 1h")
print(f"{'='*60}")
print(f"Total velas: {len(df)}")
print(f"Primer dato: {df.iloc[0]['timestamp']}")
print(f"Último dato: {df.iloc[-1]['timestamp']}")

# Verificar datos por año
df_2023 = df[df['timestamp'] >= pd.to_datetime('2023-01-01')]
df_2024 = df[df['timestamp'] >= pd.to_datetime('2024-01-01')]

print(f"\nVelas desde 2023-01-01: {len(df_2023)}")
print(f"Velas desde 2024-01-01: {len(df_2024)}")

if len(df_2024) > 0:
    print(f"\nPrimer dato 2024: {df_2024.iloc[0]['timestamp']}")
    print(f"Último dato 2024: {df_2024.iloc[-1]['timestamp']}")
else:
    print("\n⚠️ No hay datos desde 2024-01-01")
    print(f"Última fecha disponible: {df.iloc[-1]['timestamp']}")
print(f"{'='*60}\n")
