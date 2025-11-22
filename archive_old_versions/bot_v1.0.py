import os
import time
import math
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURACIÓN BOT v1.0 - MEJORAS IMPLEMENTADAS
# ============================================================================
# - Gestión de riesgo real con position sizing basado en SL
# - Control de overtrading (máx trades por día)
# - Métricas avanzadas: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor
# - Validación de equity mínimo para evitar trades microscópicos
# - Mejoras en logging y análisis de resultados
# ============================================================================

# Configuración de Exchange
API_KEY = 'lt2KQOoCSsxfHujtQNEnB9wDwCUJjZ1qOEhcUb3ws1aOamKTaOrjgzd74zyOZp2R'
API_SECRET = 'iIVQXHN5PRXPxi64AjAR42Vz2BGFvp5eBh9LFQFHKkfjzeQef7dPjMmmEjxL5CJ9'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
INITIAL_CAPITAL = 200.0  # EUR
COMMISSION = 0.001  # 0.1%

# Gestión de Riesgo MEJORADA
RISK_PERCENT = 0.02  # Arriesgar solo 2% del capital por trade
MIN_EQUITY = 10.0  # Equity mínimo para operar (evita trades microscópicos)
MAX_TRADES_PER_DAY = 2  # Máximo de entradas por día (anti-overtrading)

# Indicadores Técnicos
USE_ATR_SL = True
ATR_LENGTH = 14
ATR_MULTIPLIER = 3.5  # Reducido de 5.0 a 3.5 para mejor R:R

USE_MA_FILTER = True
MA_LENGTH = 50

USE_LONG_MA_EXIT = True
LONG_MA_LENGTH = 200

USE_TRAILING_SL_MA = True

USE_ADX_FILTER = True
ADX_LENGTH = 14
ADX_THRESHOLD = 25  # Aumentado de 20 a 25 para mayor selectividad

USE_RSI_FILTER = False  # Mantener desactivado
RSI_LENGTH = 14
RSI_THRESHOLD = 55

USE_VOLUME_FILTER = False  # Mantener desactivado
VOLUME_MA_LENGTH = 20

USE_TRAILING_TP = True
TRAILING_TP_PERCENT = 0.60  # Reducido de 0.70 a 0.60 para tomar ganancias antes

# Configuración de Simulación/Live
MODE = 'simulate'  # 'simulate' o 'live'
START_DATE = '2017-01-01'
END_DATE = '2025-12-31'
LOG_TRADES = True

# Parámetros de Descarga
FETCH_MAX_LIMIT = 1000
FETCH_SLEEP_BETWEEN = 0.5
FETCH_MAX_RETRIES = 5

# Conexión a Binance
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


def calculate_ma(df, length=MA_LENGTH):
    """Calcula Media Móvil Simple."""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=length).mean()
    df['ma'] = df['ma'].fillna(0)
    return df


def calculate_long_ma(df, length=LONG_MA_LENGTH):
    """Calcula Media Móvil Larga."""
    df = df.copy()
    df['long_ma'] = df['close'].rolling(window=length).mean()
    df['long_ma'] = df['long_ma'].fillna(0)
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


def calculate_volume_ma(df, length=VOLUME_MA_LENGTH):
    """Calcula Media Móvil de Volumen."""
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=length).mean()
    df['volume_ma'] = df['volume_ma'].fillna(0)
    return df


def calculate_metrics(trades_df, equity_curve_df, initial_capital):
    """Calcula métricas de rendimiento avanzadas."""
    metrics = {}
    
    # ROI Total
    final_equity = equity_curve_df['equity'].iloc[-1]
    metrics['initial_capital'] = initial_capital
    metrics['final_equity'] = final_equity
    metrics['total_return'] = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Número de trades
    buy_trades = trades_df[trades_df['type'] == 'buy']
    sell_trades = trades_df[trades_df['type'] == 'sell']
    metrics['total_trades'] = len(buy_trades)
    
    # Win Rate
    if len(sell_trades) > 0:
        # Calcular P&L por trade
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for i in range(len(sell_trades)):
            sell = sell_trades.iloc[i]
            # Buscar la compra correspondiente
            prev_buys = buy_trades[buy_trades.index < sell.name]
            if len(prev_buys) > 0:
                buy = prev_buys.iloc[-1]
                pnl = (sell['price'] - buy['price']) / buy['price'] * 100
                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                else:
                    losing_trades += 1
                    total_loss += abs(pnl)
        
        metrics['winning_trades'] = winning_trades
        metrics['losing_trades'] = losing_trades
        metrics['win_rate'] = (winning_trades / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
        
        # Profit Factor
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
    
    # Sharpe Ratio (asumiendo risk-free rate = 0)
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
    """Simula la estrategia de trading con mejoras v1.0."""
    equity = initial_capital
    position_size = 0.0
    entry_price = 0.0
    sl_price = 0.0
    max_price = 0.0
    trades = []
    in_position = False
    trades_today = {}  # Para controlar trades por día

    # Calcular indicadores
    df = calculate_atr(df)
    if USE_MA_FILTER or USE_TRAILING_SL_MA:
        df = calculate_ma(df)
    if USE_LONG_MA_EXIT:
        df = calculate_long_ma(df)
    if USE_ADX_FILTER:
        df = calculate_adx(df)
    if USE_RSI_FILTER:
        df = calculate_rsi(df)
    if USE_VOLUME_FILTER:
        df = calculate_volume_ma(df)

    equity_curve = []

    for i in range(1, len(df)):
        close_prev = df['close'].iloc[i-1]
        close = df['close'].iloc[i]
        open_price = df['open'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        atr = df['atr'].iloc[i]
        ma = df['ma'].iloc[i] if USE_MA_FILTER or USE_TRAILING_SL_MA else 0
        long_ma = df['long_ma'].iloc[i] if USE_LONG_MA_EXIT else 0
        adx = df['adx'].iloc[i] if USE_ADX_FILTER else 0
        rsi = df['rsi'].iloc[i] if USE_RSI_FILTER else 50
        volume = df['volume'].iloc[i]
        volume_ma = df['volume_ma'].iloc[i] if USE_VOLUME_FILTER else 0
        timestamp = df['timestamp'].iloc[i]
        date_key = timestamp.date()

        # Condición de entrada
        enter_cond = close > close_prev
        if USE_MA_FILTER:
            enter_cond = enter_cond and (close > ma)
        if USE_ADX_FILTER:
            enter_cond = enter_cond and (adx > ADX_THRESHOLD)
        if USE_RSI_FILTER:
            enter_cond = enter_cond and (rsi > RSI_THRESHOLD)
        if USE_VOLUME_FILTER:
            enter_cond = enter_cond and (volume > volume_ma)
        
        is_bearish = close < open_price
        if USE_LONG_MA_EXIT and close > long_ma:
            is_bearish = False

        # ===== GESTIÓN DE SALIDAS =====
        if in_position:
            # Trailing TP
            max_price = max(max_price, high)
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            if USE_TRAILING_TP and low <= tp_price:
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
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False
                continue

            # Trailing SL MA
            if USE_TRAILING_SL_MA and close < ma:
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
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False
                continue

            # SL ATR (actualización dinámica)
            if USE_ATR_SL:
                new_sl = close - atr * ATR_MULTIPLIER
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
                        'pnl': pnl,
                        'pnl_pct': (pnl / (position_size * entry_price)) * 100
                    })
                    position_size = 0.0
                    in_position = False
                    continue

            # Salida bearish
            if is_bearish:
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
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position_size * entry_price)) * 100
                })
                position_size = 0.0
                in_position = False

        # ===== GESTIÓN DE ENTRADAS MEJORADA =====
        if not in_position and enter_cond and equity >= MIN_EQUITY:
            # Control de trades por día
            if date_key not in trades_today:
                trades_today[date_key] = 0
            
            if trades_today[date_key] >= MAX_TRADES_PER_DAY:
                # No operar más hoy
                pass
            else:
                # Calcular position size basado en riesgo
                if USE_ATR_SL and atr > 0:
                    risk_amount = equity * RISK_PERCENT
                    stop_distance = atr * ATR_MULTIPLIER
                    qty = (risk_amount / stop_distance) * (1 - COMMISSION)
                    
                    # Limitar a no usar más del 20% del equity por trade
                    max_qty = (equity * 0.20) / close
                    qty = min(qty, max_qty)
                else:
                    # Fallback: usar 5% del equity
                    qty = (equity * 0.05) / close * (1 - COMMISSION)

                if qty > 0 and (qty * close) >= 1.0:  # Mínimo $1 por trade
                    cost = qty * close
                    if cost <= equity:
                        equity -= cost
                        position_size = qty
                        entry_price = close
                        sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                        max_price = high
                        trades.append({
                            'timestamp': timestamp,
                            'type': 'buy',
                            'price': close,
                            'qty': qty,
                            'equity': equity,
                            'reason': '',
                            'pnl': 0,
                            'pnl_pct': 0
                        })
                        in_position = True
                        trades_today[date_key] += 1

        # Registrar equity curve
        current_equity = equity + (position_size * close if in_position else 0)
        equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

    # Cerrar posición final si existe
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
            'pnl': pnl,
            'pnl_pct': (pnl / (position_size * entry_price)) * 100
        })

    print(f"\n{'='*60}")
    print(f"RESULTADOS BOT v1.0")
    print(f"{'='*60}")
    print(f"Equity final: {equity:.2f} EUR")
    print(f"Número de trades (compras): {len([t for t in trades if t['type'] == 'buy'])}")
    
    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_curve)
    
    if LOG_TRADES and len(trades) > 0:
        trades_df.to_csv('trades_log_v1.0.csv', index=False)
        equity_curve_df.to_csv('equity_curve_v1.0.csv', index=False)
        print("Logs guardados en trades_log_v1.0.csv y equity_curve_v1.0.csv")
    
    # Calcular y mostrar métricas
    if len(trades_df) > 0 and len(equity_curve_df) > 0:
        metrics = calculate_metrics(trades_df, equity_curve_df, initial_capital)
        
        print(f"\n{'='*60}")
        print(f"MÉTRICAS DE RENDIMIENTO")
        print(f"{'='*60}")
        print(f"Retorno Total: {metrics['total_return']:.2f}%")
        print(f"CAGR: {metrics['cagr']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Trades Ganadores: {metrics['winning_trades']}")
        print(f"Trades Perdedores: {metrics['losing_trades']}")
        print(f"{'='*60}\n")
        
        # Guardar métricas
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('metrics_v1.0.csv', index=False)
        print("Métricas guardadas en metrics_v1.0.csv")
    
    return trades, equity


def live_strategy():
    """Estrategia en modo live (no implementado completamente)."""
    print("Modo live no completamente implementado en v1.0")
    print("Por favor, usa MODE = 'simulate' para backtesting")
    return


if __name__ == "__main__":
    if MODE == 'simulate':
        print(f"\n{'='*60}")
        print(f"INICIANDO BACKTESTING BOT v1.0")
        print(f"{'='*60}")
        print(f"Símbolo: {SYMBOL}")
        print(f"Período: {START_DATE} a {END_DATE}")
        print(f"Capital Inicial: {INITIAL_CAPITAL} EUR")
        print(f"Riesgo por Trade: {RISK_PERCENT * 100}%")
        print(f"Máx Trades/Día: {MAX_TRADES_PER_DAY}")
        print(f"{'='*60}\n")
        
        # Descargar datos
        print("Descargando datos históricos...")
        df = fetch_ohlcv_range(SYMBOL, TIMEFRAME, since=START_DATE, until=END_DATE, 
                               max_limit=FETCH_MAX_LIMIT, sleep_between_requests=FETCH_SLEEP_BETWEEN)
        
        os.makedirs('data', exist_ok=True)
        csv_path = f"data/{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{START_DATE}_{END_DATE}_v1.0.csv"
        df.to_csv(csv_path, index=False)
        print(f"Velas guardadas en {csv_path}, {len(df)} filas\n")

        # Ejecutar simulación
        trades, final_equity = simulate_strategy(df, INITIAL_CAPITAL)
        print("\nSimulación completada.")
        
    elif MODE == 'live':
        live_strategy()
    else:
        print("Modo inválido. Usa 'simulate' o 'live'.")
