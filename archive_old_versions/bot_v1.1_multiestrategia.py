import os
import time
import math
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta

# ============================================================================
# BOT v1.1 - SISTEMA MULTI-ESTRATEGIA
# ============================================================================
# MEJORAS v1.1:
# - Sistema de 3 estrategias complementarias
# - Estrategia 1: Tendencial (la original)
# - Estrategia 2: Reversión a Media (RSI sobreventa)
# - Estrategia 3: Breakout (rotura de consolidación)
# - Selección automática según condiciones de mercado
# - Mayor frecuencia de trading manteniendo calidad
# ============================================================================

# Configuración de Exchange
API_KEY = 'lt2KQOoCSsxfHujtQNEnB9wDwCUJjZ1qOEhcUb3ws1aOamKTaOrjgzd74zyOZp2R'
API_SECRET = 'iIVQXHN5PRXPxi64AjAR42Vz2BGFvp5eBh9LFQFHKkfjzeQef7dPjMmmEjxL5CJ9'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
INITIAL_CAPITAL = 200.0
COMMISSION = 0.001

# Gestión de Riesgo
RISK_PERCENT = 0.02  # 2% por trade
MIN_EQUITY = 10.0
MAX_TRADES_PER_DAY = 3  # Aumentado a 3 para multi-estrategia

# Indicadores Comunes
ATR_LENGTH = 14
ATR_MULTIPLIER_TREND = 3.5  # Para estrategia tendencial
ATR_MULTIPLIER_REVERSAL = 2.5  # Más ajustado para reversiones
ATR_MULTIPLIER_BREAKOUT = 3.0

MA_SHORT = 20  # MA corta
MA_MID = 50    # MA media
MA_LONG = 200  # MA larga

ADX_LENGTH = 14
ADX_TRENDING = 25  # ADX > 25 = tendencia
ADX_RANGING = 20   # ADX < 20 = lateral

RSI_LENGTH = 14
RSI_OVERSOLD = 30  # Sobreventa
RSI_OVERBOUGHT = 70  # Sobrecompra

BOLLINGER_LENGTH = 20
BOLLINGER_STD = 2.0

VOLUME_MA_LENGTH = 20

# Configuración de Estrategias (activar/desactivar)
ENABLE_TREND_STRATEGY = True      # Estrategia tendencial
ENABLE_REVERSAL_STRATEGY = True   # Estrategia reversión
ENABLE_BREAKOUT_STRATEGY = True   # Estrategia breakout

# Trailing TP por estrategia
TRAILING_TP_PERCENT_TREND = 0.60
TRAILING_TP_PERCENT_REVERSAL = 0.30  # Más conservador
TRAILING_TP_PERCENT_BREAKOUT = 0.50

# Configuración
MODE = 'simulate'
START_DATE = '2017-01-01'
END_DATE = '2025-12-31'
LOG_TRADES = True

FETCH_MAX_LIMIT = 1000
FETCH_SLEEP_BETWEEN = 0.5
FETCH_MAX_RETRIES = 5

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})


def fetch_ohlcv_range(symbol, timeframe, since=None, until=None, max_limit=FETCH_MAX_LIMIT, sleep_between_requests=FETCH_SLEEP_BETWEEN):
    """Descarga todas las velas entre since y until paginando."""
    if since:
        fetch_since = exchange.parse8601(since + 'T00:00:00Z')
    else:
        fetch_since = None

    if until:
        until_dt = datetime.strptime(until, '%Y-%m-%d')
        until_ts = int(until_dt.timestamp() * 1000)
    else:
        until_ts = None

    all_ohlcv = []
    consecutive_empty = 0

    while True:
        tries = 0
        while tries <= FETCH_MAX_RETRIES:
            try:
                chunk = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=max_limit)
                break
            except ccxt.BaseError as e:
                tries += 1
                wait = min(5 * tries, 60)
                print(f"fetch_ohlcv error: {e}. retry {tries}/{FETCH_MAX_RETRIES} after {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("fetch_ohlcv raised repeated errors, aborting")

        if not chunk:
            consecutive_empty += 1
            if consecutive_empty > 2:
                break
            break

        consecutive_empty = 0
        all_ohlcv.extend(chunk)

        last_ts = chunk[-1][0]
        fetch_since = last_ts + 1

        if until_ts and fetch_since >= until_ts:
            break

        if len(chunk) < max_limit:
            break

        time.sleep(sleep_between_requests)

    if not all_ohlcv:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    if until_ts:
        df = df[df['timestamp'] < pd.to_datetime(until_ts, unit='ms')].reset_index(drop=True)

    return df


def calculate_atr(df, length=ATR_LENGTH):
    """Calcula el Average True Range."""
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low'] - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    atr = [float('nan')] * len(df)
    for i in range(length, len(df)):
        if i == length:
            atr[i] = df['tr'].iloc[i-length+1:i+1].mean()
        else:
            atr[i] = (atr[i-1] * (length - 1) + df['tr'].iloc[i]) / length
    df['atr'] = atr
    df['atr'] = df['atr'].fillna(0)
    return df


def calculate_moving_averages(df):
    """Calcula todas las medias móviles necesarias."""
    df = df.copy()
    df['ma_short'] = df['close'].rolling(window=MA_SHORT).mean()
    df['ma_mid'] = df['close'].rolling(window=MA_MID).mean()
    df['ma_long'] = df['close'].rolling(window=MA_LONG).mean()
    df['ma_short'] = df['ma_short'].fillna(0)
    df['ma_mid'] = df['ma_mid'].fillna(0)
    df['ma_long'] = df['ma_long'].fillna(0)
    return df


def calculate_adx(df, length=ADX_LENGTH):
    """Calcula el Average Directional Index."""
    df = df.copy()
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['dm_plus'] = (df['high'] - df['high'].shift(1)).where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
    df['dm_minus'] = (df['low'].shift(1) - df['low']).where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
    df['tr_sm'] = df['tr'].rolling(length).mean()
    df['di_plus'] = (df['dm_plus'].rolling(length).mean() / df['tr_sm']) * 100
    df['di_minus'] = (df['dm_minus'].rolling(length).mean() / df['tr_sm']) * 100
    df['dx'] = (abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
    df['adx'] = df['dx'].rolling(length).mean()
    df['adx'] = df['adx'].fillna(0)
    return df


def calculate_rsi(df, length=RSI_LENGTH):
    """Calcula el Relative Strength Index."""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=length).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    return df


def calculate_bollinger_bands(df, length=BOLLINGER_LENGTH, std_dev=BOLLINGER_STD):
    """Calcula Bollinger Bands."""
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(window=length).mean()
    df['bb_std'] = df['close'].rolling(window=length).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * std_dev)
    df['bb_mid'] = df['bb_mid'].fillna(0)
    df['bb_upper'] = df['bb_upper'].fillna(0)
    df['bb_lower'] = df['bb_lower'].fillna(0)
    return df


def calculate_volume_ma(df, length=VOLUME_MA_LENGTH):
    """Calcula Media Móvil de Volumen."""
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=length).mean()
    df['volume_ma'] = df['volume_ma'].fillna(0)
    return df


def detect_market_regime(adx, close, ma_short, ma_mid):
    """Detecta el régimen de mercado."""
    if adx > ADX_TRENDING and close > ma_mid:
        return 'trending_bull'
    elif adx > ADX_TRENDING and close < ma_mid:
        return 'trending_bear'
    elif adx < ADX_RANGING:
        return 'ranging'
    else:
        return 'transitioning'


def check_trend_strategy_entry(close, close_prev, adx, ma_mid, ma_long):
    """Estrategia 1: Tendencial (original mejorada)."""
    if not ENABLE_TREND_STRATEGY:
        return False
    
    # Condiciones de entrada tendencial
    bullish_candle = close > close_prev
    above_ma = close > ma_mid
    strong_trend = adx > ADX_TRENDING
    long_term_bull = close > ma_long if ma_long > 0 else True
    
    return bullish_candle and above_ma and strong_trend and long_term_bull


def check_reversal_strategy_entry(close, rsi, adx, bb_lower, ma_short, ma_mid):
    """Estrategia 2: Reversión a Media."""
    if not ENABLE_REVERSAL_STRATEGY:
        return False
    
    # Condiciones de reversión (captura sobreventa)
    oversold = rsi < RSI_OVERSOLD
    ranging_market = adx < ADX_RANGING
    near_bb_lower = close < bb_lower * 1.02 if bb_lower > 0 else False
    above_ma_short = close > ma_short * 0.98  # Ligera tolerancia
    
    return oversold and ranging_market and (near_bb_lower or True)


def check_breakout_strategy_entry(df, i, adx, volume, volume_ma):
    """Estrategia 3: Breakout de consolidación."""
    if not ENABLE_BREAKOUT_STRATEGY:
        return False
    
    if i < 20:
        return False
    
    # Buscar consolidación previa (últimos 10 días)
    lookback = 10
    recent = df.iloc[i-lookback:i]
    
    # Consolidación = rango estrecho
    high_max = recent['high'].max()
    low_min = recent['low'].min()
    range_pct = (high_max - low_min) / low_min * 100
    
    # Breakout con volumen
    current_close = df['close'].iloc[i]
    breakout = current_close > high_max
    volume_surge = volume > volume_ma * 1.5 if volume_ma > 0 else False
    
    # ADX empezando a subir (tendencia emergente)
    adx_rising = adx > 15 and adx < ADX_TRENDING
    
    # Consolidación estrecha + breakout + volumen
    is_consolidation = range_pct < 15  # Rango < 15%
    
    return is_consolidation and breakout and volume_surge and adx_rising


def calculate_metrics(trades_df, equity_curve_df, initial_capital):
    """Calcula métricas de rendimiento."""
    metrics = {}
    
    final_equity = equity_curve_df['equity'].iloc[-1]
    metrics['initial_capital'] = initial_capital
    metrics['final_equity'] = final_equity
    metrics['total_return'] = ((final_equity - initial_capital) / initial_capital) * 100
    
    buy_trades = trades_df[trades_df['type'] == 'buy']
    sell_trades = trades_df[trades_df['type'] == 'sell']
    metrics['total_trades'] = len(buy_trades)
    
    # Win Rate por estrategia
    for strategy in ['trend', 'reversal', 'breakout']:
        strategy_sells = sell_trades[sell_trades['strategy'] == strategy]
        if len(strategy_sells) > 0:
            wins = len(strategy_sells[strategy_sells['pnl'] > 0])
            metrics[f'{strategy}_trades'] = len(strategy_sells)
            metrics[f'{strategy}_win_rate'] = (wins / len(strategy_sells)) * 100
        else:
            metrics[f'{strategy}_trades'] = 0
            metrics[f'{strategy}_win_rate'] = 0
    
    # Win Rate global
    if len(sell_trades) > 0:
        winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
        losing_trades = len(sell_trades[sell_trades['pnl'] <= 0])
        total_profit = sell_trades[sell_trades['pnl'] > 0]['pnl'].sum()
        total_loss = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'].sum())
        
        metrics['winning_trades'] = winning_trades
        metrics['losing_trades'] = losing_trades
        metrics['win_rate'] = (winning_trades / len(sell_trades)) * 100
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
    else:
        metrics['winning_trades'] = 0
        metrics['losing_trades'] = 0
        metrics['win_rate'] = 0
        metrics['profit_factor'] = 0
    
    # Max Drawdown
    equity_curve_df['cummax'] = equity_curve_df['equity'].cummax()
    equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['cummax']) / equity_curve_df['cummax'] * 100
    metrics['max_drawdown'] = equity_curve_df['drawdown'].min()
    
    # Sharpe Ratio
    equity_curve_df['returns'] = equity_curve_df['equity'].pct_change()
    daily_returns = equity_curve_df['returns'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        metrics['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
    else:
        metrics['sharpe_ratio'] = 0
    
    # CAGR
    years = (equity_curve_df['timestamp'].iloc[-1] - equity_curve_df['timestamp'].iloc[0]).days / 365.25
    if years > 0:
        metrics['cagr'] = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        metrics['cagr'] = 0
    
    return metrics


def simulate_strategy(df, initial_capital):
    """Simula el sistema multi-estrategia."""
    equity = initial_capital
    position_size = 0.0
    entry_price = 0.0
    sl_price = 0.0
    max_price = 0.0
    current_strategy = None
    trades = []
    in_position = False
    trades_today = {}

    # Calcular todos los indicadores
    df = calculate_atr(df)
    df = calculate_moving_averages(df)
    df = calculate_adx(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_volume_ma(df)

    equity_curve = []

    for i in range(max(MA_LONG, BOLLINGER_LENGTH, ADX_LENGTH) + 1, len(df)):
        close_prev = df['close'].iloc[i-1]
        close = df['close'].iloc[i]
        open_price = df['open'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        atr = df['atr'].iloc[i]
        ma_short = df['ma_short'].iloc[i]
        ma_mid = df['ma_mid'].iloc[i]
        ma_long = df['ma_long'].iloc[i]
        adx = df['adx'].iloc[i]
        rsi = df['rsi'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        volume = df['volume'].iloc[i]
        volume_ma = df['volume_ma'].iloc[i]
        timestamp = df['timestamp'].iloc[i]
        date_key = timestamp.date()

        # Detectar régimen de mercado
        market_regime = detect_market_regime(adx, close, ma_short, ma_mid)

        # ===== GESTIÓN DE SALIDAS =====
        if in_position:
            # Trailing TP según estrategia
            max_price = max(max_price, high)
            if current_strategy == 'trend':
                tp_percent = TRAILING_TP_PERCENT_TREND
                atr_mult = ATR_MULTIPLIER_TREND
            elif current_strategy == 'reversal':
                tp_percent = TRAILING_TP_PERCENT_REVERSAL
                atr_mult = ATR_MULTIPLIER_REVERSAL
            else:  # breakout
                tp_percent = TRAILING_TP_PERCENT_BREAKOUT
                atr_mult = ATR_MULTIPLIER_BREAKOUT
            
            tp_price = max_price * (1 - tp_percent)
            
            # Trailing TP
            if low <= tp_price:
                sell_price = tp_price
                proceeds = position_size * sell_price * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': sell_price,
                    'qty': position_size,
                    'equity': equity,
                    'reason': 'TP',
                    'strategy': current_strategy,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False
                current_strategy = None
                continue

            # Trailing SL MA (solo para trend y breakout)
            if current_strategy in ['trend', 'breakout'] and close < ma_mid:
                sell_price = close
                proceeds = position_size * sell_price * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': sell_price,
                    'qty': position_size,
                    'equity': equity,
                    'reason': 'MA_SL',
                    'strategy': current_strategy,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False
                current_strategy = None
                continue

            # SL ATR
            new_sl = close - atr * atr_mult
            sl_price = max(sl_price, new_sl)
            if low <= sl_price:
                sell_price = sl_price
                proceeds = position_size * sell_price * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': sell_price,
                    'qty': position_size,
                    'equity': equity,
                    'reason': 'SL',
                    'strategy': current_strategy,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False
                current_strategy = None
                continue

            # Salida bearish (solo para trend)
            if current_strategy == 'trend' and close < open_price:
                sell_price = close
                proceeds = position_size * sell_price * (1 - COMMISSION)
                pnl = proceeds - (position_size * entry_price)
                equity += proceeds
                trades.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': sell_price,
                    'qty': position_size,
                    'equity': equity,
                    'reason': 'bearish',
                    'strategy': current_strategy,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False
                current_strategy = None

        # ===== GESTIÓN DE ENTRADAS =====
        if not in_position and equity >= MIN_EQUITY:
            if date_key not in trades_today:
                trades_today[date_key] = 0
            
            if trades_today[date_key] < MAX_TRADES_PER_DAY:
                # Intentar cada estrategia en orden de prioridad
                entry_strategy = None
                atr_mult = ATR_MULTIPLIER_TREND
                
                # Prioridad 1: Trend (más confiable)
                if check_trend_strategy_entry(close, close_prev, adx, ma_mid, ma_long):
                    entry_strategy = 'trend'
                    atr_mult = ATR_MULTIPLIER_TREND
                
                # Prioridad 2: Reversal (oportunidades en lateral)
                elif check_reversal_strategy_entry(close, rsi, adx, bb_lower, ma_short, ma_mid):
                    entry_strategy = 'reversal'
                    atr_mult = ATR_MULTIPLIER_REVERSAL
                
                # Prioridad 3: Breakout (menos frecuente)
                elif check_breakout_strategy_entry(df, i, adx, volume, volume_ma):
                    entry_strategy = 'breakout'
                    atr_mult = ATR_MULTIPLIER_BREAKOUT
                
                if entry_strategy:
                    # Position sizing
                    if atr > 0:
                        risk_amount = equity * RISK_PERCENT
                        stop_distance = atr * atr_mult
                        qty = (risk_amount / stop_distance) * (1 - COMMISSION)
                        max_qty = (equity * 0.20) / close
                        qty = min(qty, max_qty)
                    else:
                        qty = (equity * 0.05) / close * (1 - COMMISSION)

                    if qty > 0 and (qty * close) >= 1.0:
                        cost = qty * close
                        if cost <= equity:
                            equity -= cost
                            position_size = qty
                            entry_price = close
                            sl_price = close - atr * atr_mult
                            max_price = high
                            current_strategy = entry_strategy
                            trades.append({
                                'timestamp': timestamp,
                                'type': 'buy',
                                'price': close,
                                'qty': qty,
                                'equity': equity,
                                'reason': '',
                                'strategy': entry_strategy,
                                'pnl': 0,
                                'pnl_pct': 0
                            })
                            in_position = True
                            trades_today[date_key] += 1

        # Equity curve
        current_equity = equity + (position_size * close if in_position else 0)
        equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

    # Cerrar posición final
    if in_position:
        final_close = df['close'].iloc[-1]
        proceeds = position_size * final_close * (1 - COMMISSION)
        pnl = proceeds - (position_size * entry_price)
        equity += proceeds
        trades.append({
            'timestamp': df['timestamp'].iloc[-1],
            'type': 'sell',
            'price': final_close,
            'qty': position_size,
            'equity': equity,
            'reason': 'end',
            'strategy': current_strategy,
            'pnl': pnl,
            'pnl_pct': (pnl / (position_size * entry_price)) * 100
        })

    print(f"\n{'='*70}")
    print(f"RESULTADOS BOT v1.1 - SISTEMA MULTI-ESTRATEGIA")
    print(f"{'='*70}")
    print(f"Equity final: {equity:.2f} EUR")
    print(f"Número de trades (compras): {len([t for t in trades if t['type'] == 'buy'])}")
    
    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_curve)
    
    if LOG_TRADES and len(trades) > 0:
        trades_df.to_csv('trades_log_v1.1.csv', index=False)
        equity_curve_df.to_csv('equity_curve_v1.1.csv', index=False)
        print("Logs guardados en trades_log_v1.1.csv y equity_curve_v1.1.csv")
    
    # Métricas
    if len(trades_df) > 0 and len(equity_curve_df) > 0:
        metrics = calculate_metrics(trades_df, equity_curve_df, initial_capital)
        
        print(f"\n{'='*70}")
        print(f"MÉTRICAS GENERALES")
        print(f"{'='*70}")
        print(f"Retorno Total: {metrics['total_return']:.2f}%")
        print(f"CAGR: {metrics['cagr']:.2f}%")
        print(f"Win Rate Global: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        print(f"\n{'='*70}")
        print(f"MÉTRICAS POR ESTRATEGIA")
        print(f"{'='*70}")
        for strategy in ['trend', 'reversal', 'breakout']:
            trades_count = metrics.get(f'{strategy}_trades', 0)
            win_rate = metrics.get(f'{strategy}_win_rate', 0)
            print(f"{strategy.upper():>10}: {trades_count:>3} trades | Win Rate: {win_rate:>5.1f}%")
        print(f"{'='*70}\n")
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('metrics_v1.1.csv', index=False)
        print("Métricas guardadas en metrics_v1.1.csv")
    
    return trades, equity


if __name__ == "__main__":
    if MODE == 'simulate':
        print(f"\n{'='*70}")
        print(f"INICIANDO BACKTESTING BOT v1.1 - MULTI-ESTRATEGIA")
        print(f"{'='*70}")
        print(f"Estrategias Activas:")
        print(f"  - Tendencial: {'✓' if ENABLE_TREND_STRATEGY else '✗'}")
        print(f"  - Reversión: {'✓' if ENABLE_REVERSAL_STRATEGY else '✗'}")
        print(f"  - Breakout: {'✓' if ENABLE_BREAKOUT_STRATEGY else '✗'}")
        print(f"Símbolo: {SYMBOL} | Período: {START_DATE} to {END_DATE}")
        print(f"Capital Inicial: {INITIAL_CAPITAL} EUR")
        print(f"{'='*70}\n")
        
        print("Descargando datos históricos...")
        df = fetch_ohlcv_range(SYMBOL, TIMEFRAME, since=START_DATE, until=END_DATE, 
                               max_limit=FETCH_MAX_LIMIT, sleep_between_requests=FETCH_SLEEP_BETWEEN)
        
        os.makedirs('data', exist_ok=True)
        csv_path = f"data/{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{START_DATE}_{END_DATE}_v1.1.csv"
        df.to_csv(csv_path, index=False)
        print(f"Velas guardadas en {csv_path}, {len(df)} filas\n")

        trades, final_equity = simulate_strategy(df, INITIAL_CAPITAL)
        print("\nSimulación completada.")
    else:
        print("Modo live no implementado en v1.1")
