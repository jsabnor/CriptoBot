import os
import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import json

# ============================================================================
# BACKTESTING MULTI-ACTIVO Y MULTI-TIMEFRAME - BOT v1.0
# ============================================================================
# Prueba la estrategia v1.0 en:
# - Múltiples pares (BTC, ETH, BNB, XRP, SOL, DOGE, ADA)
# - Múltiples timeframes (1h, 4h, 1d)
# - Período: 2020-2025 (5 años)
# ============================================================================

# Configuración
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Pares a testear (top volumen en Binance)
SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'XRP/USDT',
    'SOL/USDT',
    'DOGE/USDT',
    'ADA/USDT'
]

# Timeframes a testear
TIMEFRAMES = ['1h', '4h', '1d']

# Período de backtesting (5 años para tener datos suficientes)
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'

INITIAL_CAPITAL = 200.0
COMMISSION = 0.001

# Parámetros v1.0
RISK_PERCENT = 0.02
MIN_EQUITY = 10.0
MAX_TRADES_PER_DAY = 2

ATR_LENGTH = 14
ATR_MULTIPLIER = 3.5
MA_LENGTH = 50
LONG_MA_LENGTH = 200
ADX_LENGTH = 14
ADX_THRESHOLD = 25
TRAILING_TP_PERCENT = 0.60

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})


def fetch_ohlcv_range(symbol, timeframe, since, until, max_retries=3):
    """Descarga datos históricos con manejo de errores robusto."""
    try:
        since_ts = exchange.parse8601(since + 'T00:00:00Z')
        until_dt = datetime.strptime(until, '%Y-%m-%d')
        until_ts = int(until_dt.timestamp() * 1000)
    except Exception as e:
        print(f"Error parseando fechas: {e}")
        return pd.DataFrame()

    all_ohlcv = []
    fetch_since = since_ts
    
    while fetch_since < until_ts:
        for attempt in range(max_retries):
            try:
                chunk = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=1000)
                if not chunk:
                    break
                all_ohlcv.extend(chunk)
                fetch_since = chunk[-1][0] + 1
                if len(chunk) < 1000:
                    break
                time.sleep(0.2)  # Rate limiting
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error descargando {symbol} {timeframe}: {e}")
                    return pd.DataFrame()
                time.sleep(2)
    
    if not all_ohlcv:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] < pd.to_datetime(until_ts, unit='ms')].reset_index(drop=True)
    
    return df


def calculate_indicators(df):
    """Calcula todos los indicadores necesarios para v1.0."""
    df = df.copy()
    
    # ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low'] - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    
    # ATR (Wilder's Smoothing)
    alpha = 1 / ATR_LENGTH
    df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean().fillna(0)
    
    # Moving Averages
    df['ma'] = df['close'].rolling(window=MA_LENGTH).mean().fillna(0)
    df['long_ma'] = df['close'].rolling(window=LONG_MA_LENGTH).mean().fillna(0)
    
    # ADX (Standard Welles Wilder - Matching TradingView)
    df['dm_plus'] = (df['high'] - df['high'].shift(1)).where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
    df['dm_minus'] = (df['low'].shift(1) - df['low']).where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
    
    # Smoothing alpha for ADX
    alpha_adx = 1 / ADX_LENGTH
    
    # Smoothed TR and DM
    df['tr_sm'] = df['tr'].ewm(alpha=alpha_adx, adjust=False).mean()
    df['dm_plus_sm'] = df['dm_plus'].ewm(alpha=alpha_adx, adjust=False).mean()
    df['dm_minus_sm'] = df['dm_minus'].ewm(alpha=alpha_adx, adjust=False).mean()
    
    # DI+ and DI-
    df['di_plus'] = (df['dm_plus_sm'] / df['tr_sm']) * 100
    df['di_minus'] = (df['dm_minus_sm'] / df['tr_sm']) * 100
    
    # DX
    df['dx'] = (abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
    
    # ADX (Smoothed DX)
    df['adx'] = df['dx'].ewm(alpha=alpha_adx, adjust=False).mean().fillna(0)
    
    return df


def simulate_v1_0(df, initial_capital):
    """Simula la estrategia v1.0."""
    if len(df) < LONG_MA_LENGTH:
        return {
            'final_equity': initial_capital,
            'total_return': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'cagr': 0,
            'error': 'Insufficient data'
        }
    
    df = calculate_indicators(df)
    
    equity = initial_capital
    position_size = 0.0
    entry_price = 0.0
    sl_price = 0.0
    max_price = 0.0
    in_position = False
    trades = []
    equity_curve = []
    trades_today = {}
    
    for i in range(LONG_MA_LENGTH + 1, len(df)):
        close_prev = df['close'].iloc[i-1]
        close = df['close'].iloc[i]
        open_price = df['open'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        atr = df['atr'].iloc[i]
        ma = df['ma'].iloc[i]
        long_ma = df['long_ma'].iloc[i]
        adx = df['adx'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        date_key = timestamp.date()
        
        # Gestión de salidas
        if in_position:
            max_price = max(max_price, high)
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            
            # Trailing TP
            if low <= tp_price:
                proceeds = position_size * tp_price * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({'pnl': pnl})
                position_size = 0.0
                in_position = False
                continue
            
            # Trailing SL MA
            if close < ma:
                proceeds = position_size * close * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({'pnl': pnl})
                position_size = 0.0
                in_position = False
                continue
            
            # SL ATR
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl)
            if low <= sl_price:
                proceeds = position_size * sl_price * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({'pnl': pnl})
                position_size = 0.0
                in_position = False
                continue
            
            # Bearish exit
            if close < open_price and close < long_ma:
                proceeds = position_size * close * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({'pnl': pnl})
                position_size = 0.0
                in_position = False
        
        # Gestión de entradas
        if not in_position and equity >= MIN_EQUITY:
            if date_key not in trades_today:
                trades_today[date_key] = 0
            
            if trades_today[date_key] < MAX_TRADES_PER_DAY:
                # Estrategia v1.0: Trend
                enter_cond = (close > close_prev and close > ma and 
                             adx > ADX_THRESHOLD and 
                             (close > long_ma if long_ma > 0 else True))
                
                if enter_cond and atr > 0:
                    risk_amount = equity * RISK_PERCENT
                    stop_distance = atr * ATR_MULTIPLIER
                    qty = (risk_amount / stop_distance) * (1 - COMMISSION)
                    max_qty = (equity * 0.20) / close
                    qty = min(qty, max_qty)
                    
                    if qty > 0 and (qty * close) >= 1.0:
                        cost = qty * close
                        if cost <= equity:
                            equity -= cost
                            position_size = qty
                            entry_price = close
                            sl_price = close - atr * ATR_MULTIPLIER
                            max_price = high
                            in_position = True
                            trades_today[date_key] += 1
        
        current_equity = equity + (position_size * close if in_position else 0)
        equity_curve.append(current_equity)
    
    # Cerrar posición final
    if in_position:
        final_close = df['close'].iloc[-1]
        proceeds = position_size * final_close * (1 - COMMISSION)
        pnl = proceeds - (position_size * entry_price)
        equity += proceeds
        trades.append({'pnl': pnl})
    
    # Calcular métricas
    if len(trades) == 0:
        return {
            'final_equity': equity,
            'total_return': ((equity - initial_capital) / initial_capital) * 100,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'cagr': 0
        }
    
    wins = len([t for t in trades if t['pnl'] > 0])
    win_rate = (wins / len(trades)) * 100 if len(trades) > 0 else 0
    
    total_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
    total_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    # Max Drawdown
    equity_series = pd.Series(equity_curve)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax * 100
    max_dd = drawdown.min()
    
    # Sharpe Ratio
    returns = equity_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(equity_curve) / len(df))
    else:
        sharpe = 0
    
    # CAGR
    years = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days / 365.25
    cagr = ((equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    return {
        'final_equity': equity,
        'total_return': ((equity - initial_capital) / initial_capital) * 100,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'cagr': cagr
    }


def run_multi_test():
    """Ejecuta backtesting en múltiples pares y timeframes."""
    results = []
    
    total_tests = len(SYMBOLS) * len(TIMEFRAMES)
    current_test = 0
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING MULTI-ACTIVO Y MULTI-TIMEFRAME - BOT v1.0")
    print(f"{'='*80}")
    print(f"Período: {START_DATE} a {END_DATE}")
    print(f"Pares: {len(SYMBOLS)} | Timeframes: {len(TIMEFRAMES)} | Total: {total_tests} tests")
    print(f"{'='*80}\n")
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing {symbol} {timeframe}...", end=' ')
            
            try:
                # Descargar datos
                df = fetch_ohlcv_range(symbol, timeframe, START_DATE, END_DATE)
                
                if len(df) < LONG_MA_LENGTH:
                    print(f"❌ Datos insuficientes")
                    results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': 'Insufficient data',
                        **{k: 0 for k in ['final_equity', 'total_return', 'total_trades', 
                                          'win_rate', 'profit_factor', 'max_drawdown', 
                                          'sharpe_ratio', 'cagr']}
                    })
                    continue
                
                # Simular
                metrics = simulate_v1_0(df, INITIAL_CAPITAL)
                
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'candles': len(df),
                    **metrics
                }
                results.append(result)
                
                print(f"✓ ROI: {metrics['total_return']:.1f}% | Trades: {metrics['total_trades']} | WR: {metrics['win_rate']:.1f}%")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                results.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'error': str(e),
                    **{k: 0 for k in ['final_equity', 'total_return', 'total_trades', 
                                      'win_rate', 'profit_factor', 'max_drawdown', 
                                      'sharpe_ratio', 'cagr', 'candles']}
                })
    
    # Guardar resultados
    df_results = pd.DataFrame(results)
    df_results.to_csv('backtest_multi_results.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS GUARDADOS EN: backtest_multi_results.csv")
    print(f"{'='*80}\n")
    
    # Análisis rápido
    print("\n=== RANKING POR ROI ===")
    df_valid = df_results[df_results['total_trades'] > 0].copy()
    df_valid = df_valid.sort_values('total_return', ascending=False)
    print(df_valid[['symbol', 'timeframe', 'total_return', 'total_trades', 'win_rate', 'sharpe_ratio']].head(10).to_string(index=False))
    
    print("\n\n=== RANKING POR SHARPE RATIO ===")
    df_valid = df_valid.sort_values('sharpe_ratio', ascending=False)
    print(df_valid[['symbol', 'timeframe', 'total_return', 'sharpe_ratio', 'profit_factor']].head(10).to_string(index=False))
    
    # Estadísticas por timeframe
    print("\n\n=== ESTADÍSTICAS POR TIMEFRAME ===")
    for tf in TIMEFRAMES:
        tf_data = df_results[df_results['timeframe'] == tf]
        if len(tf_data) > 0:
            print(f"\n{tf}:")
            print(f"  ROI Promedio: {tf_data['total_return'].mean():.2f}%")
            print(f"  Win Rate Promedio: {tf_data['win_rate'].mean():.2f}%")
            print(f"  Sharpe Promedio: {tf_data['sharpe_ratio'].mean():.2f}")
            print(f"  Trades Promedio: {tf_data['total_trades'].mean():.0f}")
    
    # Estadísticas por símbolo
    print("\n\n=== ESTADÍSTICAS POR SÍMBOLO ===")
    for sym in SYMBOLS:
        sym_data = df_results[df_results['symbol'] == sym]
        if len(sym_data) > 0:
            print(f"\n{sym}:")
            print(f"  ROI Promedio: {sym_data['total_return'].mean():.2f}%")
            print(f"  Win Rate Promedio: {sym_data['win_rate'].mean():.2f}%")
            print(f"  Mejor TF: {sym_data.loc[sym_data['total_return'].idxmax(), 'timeframe']}")
    
    return df_results


if __name__ == "__main__":
    print("\n🚀 Iniciando backtesting multi-activo y multi-timeframe...")
    print("⏳ Esto puede tomar varios minutos...")
    
    results_df = run_multi_test()
    
    print("\n✅ Backtesting completado!")
    print(f"\n📊 Resultados guardados en: backtest_multi_results.csv")
    print(f"📈 Total de tests ejecutados: {len(results_df)}")
