
import pandas as pd
import numpy as np
from data_cache import DataCache
from datetime import datetime, timedelta
import os
import argparse

# Configuración por defecto
DEFAULT_START_DATE = datetime.now() - timedelta(days=1)
SYMBOLS = ['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']
TIMEFRAME = '4h'

# Configuración de Estrategias
ADX_CONFIG = {
    'ADX_THRESHOLD': 25,
    'ATR_MULTIPLIER': 2.0,
    'RISK_PERCENT': 0.02,
    'TRAILING_TP_PERCENT': 0.015
}

EMA_CONFIG = {
    'EMA_FAST': 15,
    'EMA_SLOW': 30,
    'RISK_PERCENT': 0.02
}

def calculate_adx_indicators(df):
    """Calcula indicadores para estrategia ADX"""
    df = df.copy()
    
    # ATR
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # ADX
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/14).mean() / df['atr'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # MAs
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    return df

def calculate_ema_indicators(df):
    """Calcula indicadores para estrategia EMA"""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # ATR para SL
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    return df

def simulate_trades(df, strategy_type, symbol):
    """Simula trades y retorna lista de operaciones"""
    trades = []
    position = None
    
    # Iterar sobre el DataFrame
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        timestamp = curr['timestamp']
        close = curr['close']
        high = curr['high']
        low = curr['low']
        
        # --- GESTIÓN DE POSICIÓN ABIERTA ---
        if position:
            # Actualizar max price para Trailing TP
            position['max_price'] = max(position['max_price'], high)
            
            exit_reason = None
            exit_price = 0
            
            if strategy_type == 'ADX':
                # Trailing TP
                tp_price = position['max_price'] * (1 - ADX_CONFIG['TRAILING_TP_PERCENT'])
                if low <= tp_price:
                    exit_reason = 'TP (Trailing)'
                    exit_price = tp_price
                
                # Stop Loss MA (Cierre por debajo de MA50)
                elif close < curr['ma50']:
                    exit_reason = 'SL (MA50)'
                    exit_price = close
                    
            elif strategy_type == 'EMA':
                # SL Fijo (2 ATR)
                if low <= position['sl_price']:
                    exit_reason = 'SL (Fijo)'
                    exit_price = position['sl_price']
                
                # Take Profit (2:1)
                elif high >= position['tp_price']:
                    exit_reason = 'TP (2:1)'
                    exit_price = position['tp_price']
                    
                # Cruce Bajista (Salida anticipada)
                elif curr['ema_fast'] < curr['ema_slow']:
                    exit_reason = 'Cruce Bajista'
                    exit_price = close
            
            # Ejecutar Salida
            if exit_reason:
                pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                trades.append({
                    'symbol': symbol,
                    'strategy': strategy_type,
                    'type': 'SELL',
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'reason': exit_reason
                })
                position = None
                continue

        # --- GESTIÓN DE ENTRADA ---
        if position is None:
            entry_signal = False
            sl_price = 0
            tp_price = 0
            
            if strategy_type == 'ADX':
                # Condiciones ADX
                if (curr['adx'] > ADX_CONFIG['ADX_THRESHOLD'] and 
                    curr['close'] > curr['ma50'] and 
                    curr['close'] > curr['ma200'] and
                    curr['plus_di'] > curr['minus_di']):
                    entry_signal = True
                    
            elif strategy_type == 'EMA':
                # Condiciones EMA (Cruce Alcista)
                if (curr['ema_fast'] > curr['ema_slow'] and 
                    prev['ema_fast'] <= prev['ema_slow']):
                    entry_signal = True
                    sl_price = close - (curr['atr'] * 2)
                    risk = close - sl_price
                    tp_price = close + (risk * 2)

            if entry_signal:
                position = {
                    'entry_time': timestamp,
                    'entry_price': close,
                    'max_price': close,
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
                # No añadimos a 'trades' todavía, solo cuando cierre
                
    # Si queda posición abierta al final
    if position:
        curr = df.iloc[-1]
        pnl_pct = ((curr['close'] - position['entry_price']) / position['entry_price']) * 100
        trades.append({
            'symbol': symbol,
            'strategy': strategy_type,
            'type': 'OPEN',
            'entry_time': position['entry_time'],
            'exit_time': 'Abierta',
            'entry_price': position['entry_price'],
            'exit_price': curr['close'],
            'pnl_pct': pnl_pct,
            'reason': 'En Curso'
        })
        
    return trades

def main():
    parser = argparse.ArgumentParser(description='Verificar señales históricas de los bots')
    parser.add_argument('--days', type=int, default=2, help='Días hacia atrás para analizar (default: 2)')
    parser.add_argument('--symbol', type=str, help='Símbolo específico a analizar (opcional)')
    args = parser.parse_args()

    start_time = datetime.now() - timedelta(days=args.days)
    symbols_to_check = [args.symbol] if args.symbol else SYMBOLS
    
    cache = DataCache()
    all_trades = []
    
    print(f"\n🔍 VERIFICADOR DE SEÑALES Y OPERACIONES")
    print(f"==================================================")
    print(f"📅 Desde: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"⏱️ Timeframe: {TIMEFRAME}")
    print(f"==================================================\n")
    
    for symbol in symbols_to_check:
        print(f"Analizando {symbol}...")
        
        df = cache.get_data(symbol, TIMEFRAME)
        if df is None or df.empty:
            print(f"  ❌ No hay datos")
            continue
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filtrar datos desde start_time (con margen para indicadores)
        # Necesitamos datos previos para calcular indicadores correctamente
        # Así que calculamos indicadores sobre TODO y luego filtramos los trades
        
        # Calcular indicadores
        df_adx = calculate_adx_indicators(df)
        df_ema = calculate_ema_indicators(df)
        
        # Simular trades
        # Filtramos df para la simulación para no simular desde el inicio de los tiempos
        # pero dando un margen de 50 velas para MAs
        sim_start_idx = df[df['timestamp'] >= start_time].index[0] if not df[df['timestamp'] >= start_time].empty else 0
        sim_start_idx = max(0, sim_start_idx - 50) 
        
        df_sim_adx = df_adx.iloc[sim_start_idx:].reset_index(drop=True)
        df_sim_ema = df_ema.iloc[sim_start_idx:].reset_index(drop=True)
        
        trades_adx = simulate_trades(df_sim_adx, 'ADX', symbol)
        trades_ema = simulate_trades(df_sim_ema, 'EMA', symbol)
        
        # Filtrar solo trades que ocurrieron DENTRO del rango solicitado
        # (La simulación empezó un poco antes para estabilidad)
        trades_adx = [t for t in trades_adx if t['entry_time'] >= start_time]
        trades_ema = [t for t in trades_ema if t['entry_time'] >= start_time]
        
        all_trades.extend(trades_adx)
        all_trades.extend(trades_ema)

    # --- IMPRIMIR TABLA DE RESULTADOS ---
    print(f"\n{'='*100}")
    print(f"{'ESTRATEGIA':<10} | {'SÍMBOLO':<10} | {'TIPO':<8} | {'ENTRADA':<16} | {'SALIDA':<16} | {'PRECIO ENT.':<12} | {'PNL %':<8} | {'RAZÓN'}")
    print(f"{'-'*100}")
    
    if not all_trades:
        print(f"{'SIN OPERACIONES DETECTADAS':^100}")
    else:
        # Ordenar por fecha de entrada
        all_trades.sort(key=lambda x: x['entry_time'])
        
        for t in all_trades:
            entry_str = t['entry_time'].strftime('%m-%d %H:%M')
            exit_str = t['exit_time'].strftime('%m-%d %H:%M') if isinstance(t['exit_time'], datetime) else t['exit_time']
            pnl_str = f"{t['pnl_pct']:+.2f}%"
            pnl_color = "🟢" if t['pnl_pct'] > 0 else "🔴"
            
            print(f"{t['strategy']:<10} | {t['symbol']:<10} | {t['type']:<8} | {entry_str:<16} | {exit_str:<16} | ${t['entry_price']:<11.2f} | {pnl_str:<8} | {t['reason']}")

    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
