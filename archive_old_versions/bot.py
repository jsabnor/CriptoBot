import os
import time
import pandas as pd
import ccxt
from datetime import datetime, timedelta
import pandas_ta as pta  # Alternative if ta-lib not installed: pip install pandas_ta

# Configuración inicial
API_KEY = 'lt2KQOoCSsxfHujtQNEnB9wDwCUJjZ1qOEhcUb3ws1aOamKTaOrjgzd74zyOZp2R'  # Reemplaza con tu API key de Binance
API_SECRET = 'iIVQXHN5PRXPxi64AjAR42Vz2BGFvp5eBh9LFQFHKkfjzeQef7dPjMmmEjxL5CJ9'  # Reemplaza con tu API secret
SYMBOL = 'DOGE/EUR'  # Par de trading, ej. BTC/USDT
TIMEFRAME = '1w'  # Temporalidad: '1m', '5m', '1h', '1d', '1w', etc.
INITIAL_CAPITAL = 200.0  # Capital inicial en USDT (para simulación)
COMMISSION = 0.001  # Comisión por trade (0.1% = 0.001)
USE_ATR_SL = True  # Usar SL basado en ATR
ATR_LENGTH = 14
ATR_MULTIPLIER = 2.0
ALLOW_REENTRY = True  # Permitir reentrada tras cierre

# Modo: 'live' para ejecución real, 'simulate' para backtest
MODE = 'simulate'  # Cambia a 'live' para trading real
START_DATE = '2023-01-01'  # Para simulación: fecha inicio (YYYY-MM-DD)
END_DATE = '2025-11-12'  # Para simulación: fecha fin (YYYY-MM-DD) o actual

# Conexión a Binance via ccxt
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},  # Cambia a 'future' si usas futuros
})

def fetch_ohlcv(symbol, timeframe, since=None, limit=None):
    """Obtiene datos OHLCV históricos o en vivo."""
    if since:
        since = exchange.parse8601(since + 'T00:00:00Z')
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_atr(df, length=ATR_LENGTH):
    """Calcula ATR usando pandas_ta."""
    df['atr'] = pta.atr(high=df['high'], low=df['low'], close=df['close'], length=length)
    return df

def simulate_strategy(df, initial_capital):
    """Simulación (backtest) de la estrategia."""
    equity = initial_capital
    position_size = 0.0  # Cantidad en crypto
    sl_price = 0.0
    trades = []
    in_position = False

    df = calculate_atr(df)  # Agrega ATR

    for i in range(1, len(df)):
        close_prev = df['close'].iloc[i-1]
        close = df['close'].iloc[i]
        open_price = df['open'].iloc[i]
        atr = df['atr'].iloc[i]
        timestamp = df['timestamp'].iloc[i]

        enter_cond = close > close_prev
        is_bearish = close < open_price

        # Cálculo de SL
        if USE_ATR_SL and in_position:
            sl_price = max(sl_price, close - atr * ATR_MULTIPLIER)  # Trailing SL (actualiza si sube)

        # Salida por vela bajista
        if in_position and is_bearish:
            sell_price = close  # Asume salida en close (simplificado)
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'type': 'sell', 'price': sell_price, 'qty': position_size, 'timestamp': timestamp, 'reason': 'bearish'})
            position_size = 0.0
            in_position = False

        # Salida por SL (chequea si low <= SL)
        if in_position and USE_ATR_SL and df['low'].iloc[i] <= sl_price:
            sell_price = sl_price  # Asume hit en SL
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'type': 'sell', 'price': sell_price, 'qty': position_size, 'timestamp': timestamp, 'reason': 'SL'})
            position_size = 0.0
            in_position = False

        # Entrada all-in
        can_enter = enter_cond and (not in_position or ALLOW_REENTRY)
        if can_enter:
            qty = equity / close * (1 - COMMISSION)  # All-in, ajustado por comisión
            if qty > 0:
                position_size += qty
                equity -= qty * close  # Resta el costo (queda en 0 aprox, pero por comisiones)
                sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                trades.append({'type': 'buy', 'price': close, 'qty': qty, 'timestamp': timestamp})
                in_position = True

    # Cierra posición final si abierta
    if in_position:
        final_close = df['close'].iloc[-1]
        proceeds = position_size * final_close * (1 - COMMISSION)
        equity += proceeds
        trades.append({'type': 'sell', 'price': final_close, 'qty': position_size, 'timestamp': df['timestamp'].iloc[-1], 'reason': 'end'})

    print(f"Equity final: {equity:.2f} USDT")
    print(f"Número de trades: {len(trades) // 2}")
    return trades, equity

def live_strategy():
    """Ejecución en vivo: Monitorea y tradea en real time."""
    print("Iniciando modo live... Presiona Ctrl+C para detener.")
    in_position = False
    position_size = 0.0
    entry_price = 0.0
    sl_price = 0.0

    while True:
        # Fetch última barra completa
        df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=2)
        df = calculate_atr(df)

        close_prev = df['close'].iloc[0]
        close = df['close'].iloc[1]
        open_price = df['open'].iloc[1]
        low = df['low'].iloc[1]
        atr = df['atr'].iloc[1]
        is_bearish = close < open_price
        enter_cond = close > close_prev

        # Actualiza SL si en posición
        if in_position and USE_ATR_SL:
            sl_price = max(sl_price, close - atr * ATR_MULTIPLIER)

        # Chequea SL (en live, usa orders condicionales, pero simplificado aquí)
        if in_position and USE_ATR_SL:
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            if current_price <= sl_price:
                # Cierra posición
                order = exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"SL hit: Vendido {position_size} a {current_price}")
                in_position = False
                position_size = 0.0

        # Salida por vela bajista (al cierre de barra)
        if in_position and is_bearish:
            order = exchange.create_market_sell_order(SYMBOL, position_size)
            print(f"Vela bajista: Vendido {position_size} a {close}")
            in_position = False
            position_size = 0.0

        # Entrada
        can_enter = enter_cond and (not in_position or ALLOW_REENTRY)
        if can_enter:
            balance = exchange.fetch_balance()['USDT']['free']
            qty = balance / close * (1 - COMMISSION)  # All-in
            if qty > 0:
                order = exchange.create_market_buy_order(SYMBOL, qty)
                position_size += qty
                sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                print(f"Comprado {qty} a {close}")
                in_position = True

        # Espera hasta próxima barra
        timeframe_seconds = exchange.parse_timeframe(TIMEFRAME) * 1000 // 1000  # ms a seg
        time.sleep(timeframe_seconds)

if __name__ == "__main__":
    if MODE == 'simulate':
        # Para simulación: fetch datos históricos
        start_timestamp = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_timestamp = datetime.strptime(END_DATE, '%Y-%m-%d') if END_DATE else datetime.now()
        days = (end_timestamp - start_timestamp).days
        limit = days + ATR_LENGTH + 1  # Suficiente para timeframe '1d'
        df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=START_DATE, limit=limit)
        trades, final_equity = simulate_strategy(df, INITIAL_CAPITAL)
        print("Simulación completada.")
    elif MODE == 'live':
        live_strategy()
    else:
        print("Modo inválido. Usa 'simulate' o 'live'.")