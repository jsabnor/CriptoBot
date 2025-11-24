import pandas as pd
import itertools
from data_cache import DataCache
from backtest_multi import simulate_v1_0
import time

# Configuración de Optimización
SYMBOLS = ['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']
TIMEFRAME = '4h'
INITIAL_CAPITAL = 200.0

# Rangos de Parámetros (Grid Search)
PARAM_GRID = {
    'ADX_THRESHOLD': [20, 25, 30],
    'ATR_MULTIPLIER': [2.5, 3.0, 3.5, 4.0],
    'RISK_PERCENT': [0.02, 0.03, 0.04],
    'TRAILING_TP_PERCENT': [0.5, 0.6, 0.7]
}

def run_optimization():
    print(f"🚀 Iniciando Optimización de Estrategia (Grid Search)")
    print(f"📊 Pares: {SYMBOLS}")
    print(f"⏱️ Timeframe: {TIMEFRAME}")
    
    # Cargar datos en memoria (SOLO CACHÉ)
    cache = DataCache()
    data_cache = {}
    
    print("\n📥 Cargando datos desde caché...")
    for symbol in SYMBOLS:
        # force_update=False asegura que NO descargue nada nuevo
        df = cache.get_data(symbol, TIMEFRAME, force_update=False)
        if df is not None and len(df) > 200:
            data_cache[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} velas cargadas")
        else:
            print(f"  ❌ {symbol}: No hay datos suficientes en caché")
    
    if not data_cache:
        print("❌ No hay datos para optimizar. Ejecuta primero backtest_multi.py para descargar caché.")
        return

    # Generar combinaciones
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n🔄 Probando {len(combinations)} combinaciones...")
    print(f"   Total simulaciones: {len(combinations) * len(data_cache)}")
    
    results = []
    start_time = time.time()
    
    for i, config in enumerate(combinations):
        if i % 10 == 0:
            print(f"   Progreso: {i}/{len(combinations)}...")
            
        # Configuración base + parámetros variables
        full_config = {
            'ATR_LENGTH': 14,
            'MA_LENGTH': 50,
            'LONG_MA_LENGTH': 200,
            'ADX_LENGTH': 14,
            'MIN_EQUITY': 10.0,
            'MAX_TRADES_PER_DAY': 2,
            'COMMISSION': 0.001,
            **config
        }
        
        # Ejecutar simulación para cada par
        total_roi = 0
        total_dd = 0
        total_sharpe = 0
        valid_pairs = 0
        
        for symbol, df in data_cache.items():
            metrics = simulate_v1_0(df, INITIAL_CAPITAL, full_config)
            
            if metrics['total_trades'] > 0:
                total_roi += metrics['total_return']
                total_dd += metrics['max_drawdown']
                total_sharpe += metrics['sharpe_ratio']
                valid_pairs += 1
        
        if valid_pairs > 0:
            avg_roi = total_roi / valid_pairs
            avg_dd = total_dd / valid_pairs
            avg_sharpe = total_sharpe / valid_pairs
            
            # Score personalizado: ROI / abs(DD) (Reward to Risk)
            score = avg_roi / abs(avg_dd) if avg_dd != 0 else 0
            
            results.append({
                **config,
                'avg_roi': avg_roi,
                'avg_dd': avg_dd,
                'avg_sharpe': avg_sharpe,
                'score': score
            })
    
    elapsed = time.time() - start_time
    print(f"\n✅ Optimización completada en {elapsed:.1f}s")
    
    # Guardar resultados
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False)
    df_results.to_csv('optimization_results.csv', index=False)
    
    print(f"\n🏆 MEJORES 5 CONFIGURACIONES (por Score):")
    print(df_results.head(5).to_string(index=False))
    
    print(f"\n💰 MEJORES 5 POR ROI:")
    print(df_results.sort_values('avg_roi', ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    run_optimization()
