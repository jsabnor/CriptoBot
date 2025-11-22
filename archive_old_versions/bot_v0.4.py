import os
import time
import pandas as pd
import ccxt
from datetime import datetime, timedelta

# Configuración inicial
API_KEY = 'lt2KQOoCSsxfHujtQNEnB9wDwCUJjZ1qOEhcUb3ws1aOamKTaOrjgzd74zyOZp2R'  # Reemplaza con tu API key de Binance
API_SECRET = 'iIVQXHN5PRXPxi64AjAR42Vz2BGFvp5eBh9LFQFHKkfjzeQef7dPjMmmEjxL5CJ9'  # Reemplaza con tu API secret
SYMBOL = 'DOGE/EUR'  # Cambiado a DOGE/EUR
TIMEFRAME = '1w'  # Temporalidad
INITIAL_CAPITAL = 200.0  # Capital inicial en EUR
COMMISSION = 0.001  # Comisión por trade (0.1% = 0.001)
USE_ATR_SL = True  # Usar SL basado en ATR
ATR_LENGTH = 14
ATR_MULTIPLIER = 3.5  # Aumentado ligeramente para DOGE volatil
ALLOW_REENTRY = True  # Permitir reentrada
USE_MA_FILTER = True  # Filtro MA para entradas
MA_LENGTH = 50
USE_RSI_FILTER = False  # Desactivado para DOGE (prueba True si overfilters)
RSI_LENGTH = 14
RSI_THRESHOLD = 50
USE_VOLUME_FILTER = False  # Opcional
VOLUME_MA_LENGTH = 20
USE_TRAILING_TP = True  # Trailing take profit
TRAILING_TP_PERCENT = 0.20  # 20% para hold más en pumps de DOGE
RISK_PERCENT = 1.0  # 1.0 para all-in (como v0.2)

# Modo: 'live' para ejecución real, 'simulate' para backtest
MODE = 'simulate'  # Cambia a 'live' para trading real
START_DATE = '2023-01-01'  # Para simulación: fecha inicio (YYYY-MM-DD)
END_DATE = '2025-11-12'  # Para simulación: fecha fin (YYYY-MM-DD) o actual
LOG_TRADES = True  # Guardar trades en CSV

# Conexión a Binance via ccxt
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},  # Cambia a 'future' si usas futuros
})

def fetch_ohlcv(symbol, timeframe, since=None, limit=None):
    if since:
        since = exchange.parse8601(since + 'T00:00:00Z')
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_atr(df, length=ATR_LENGTH):
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
    df['ma'] = df['close'].rolling(window=length).mean()
    df['ma'] = df['ma'].fillna(0)
    return df

def calculate_rsi(df, length=RSI_LENGTH):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=length).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    return df

def calculate_volume_ma(df, length=VOLUME_MA_LENGTH):
    df['volume_ma'] = df['volume'].rolling(window=length).mean()
    df['volume_ma'] = df['volume_ma'].fillna(0)
    return df

def simulate_strategy(df, initial_capital):
    equity = initial_capital
    position_size = 0.0
    sl_price = 0.0
    max_price = 0.0
    trades = []
    in_position = False

    df = calculate_atr(df)
    if USE_MA_FILTER:
        df = calculate_ma(df)
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
        ma = df['ma'].iloc[i] if USE_MA_FILTER else 0
        rsi = df['rsi'].iloc[i] if USE_RSI_FILTER else 50
        volume = df['volume'].iloc[i]
        volume_ma = df['volume_ma'].iloc[i] if USE_VOLUME_FILTER else 0
        timestamp = df['timestamp'].iloc[i]

        enter_cond = close > close_prev
        if USE_MA_FILTER:
            enter_cond = enter_cond and (close > ma)
        if USE_RSI_FILTER:
            enter_cond = enter_cond and (rsi > RSI_THRESHOLD)
        if USE_VOLUME_FILTER:
            enter_cond = enter_cond and (volume > volume_ma)
        is_bearish = close < open_price

        # Trailing TP
        if in_position:
            max_price = max(max_price, high)
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            if USE_TRAILING_TP and low <= tp_price:  # Chequea low para simular intra-bar
                sell_price = tp_price
                proceeds = position_size * sell_price * (1 - COMMISSION)
                equity += proceeds
                trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'TP', 'equity': equity})
                position_size = 0.0
                in_position = False
                continue

        # SL update
        if USE_ATR_SL and in_position:
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl) if new_sl > sl_price else sl_price  # Solo mejora

        # Salida bearish
        if in_position and is_bearish:
            sell_price = close
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'bearish', 'equity': equity})
            position_size = 0.0
            in_position = False

        # Salida SL
        if in_position and USE_ATR_SL and low <= sl_price:
            sell_price = sl_price
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'SL', 'equity': equity})
            position_size = 0.0
            in_position = False

        # Entrada
        can_enter = enter_cond and (not in_position or ALLOW_REENTRY)
        if can_enter:
            risk_equity = equity * RISK_PERCENT
            qty = risk_equity / close * (1 - COMMISSION)
            if qty > 0:
                position_size += qty
                equity -= qty * close
                sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                max_price = high  # Usa high para reset
                trades.append({'timestamp': timestamp, 'type': 'buy', 'price': close, 'qty': qty, 'equity': equity})
                in_position = True

        equity_curve.append({'timestamp': timestamp, 'equity': equity + (position_size * close if in_position else 0)})

    if in_position:
        final_close = df['close'].iloc[-1]
        proceeds = position_size * final_close * (1 - COMMISSION)
        equity += proceeds
        trades.append({'timestamp': df['timestamp'].iloc[-1], 'type': 'sell', 'price': final_close, 'qty': position_size, 'reason': 'end', 'equity': equity})

    print(f"Equity final: {equity:.2f} EUR")
    print(f"Número de trades: {len([t for t in trades if t['type'] == 'buy'])}")

    if LOG_TRADES:
        pd.DataFrame(trades).to_csv('trades_log.csv', index=False)
        pd.DataFrame(equity_curve).to_csv('equity_curve.csv', index=False)
        print("Logs guardados en trades_log.csv y equity_curve.csv")

    return trades, equity

def live_strategy():
    print("Iniciando modo live... Presiona Ctrl+C para detener.")
    in_position = False
    position_size = 0.0
    sl_price = 0.0
    max_price = 0.0

    while True:
        limit = max(ATR_LENGTH, MA_LENGTH if USE_MA_FILTER else 0, RSI_LENGTH if USE_RSI_FILTER else 0, VOLUME_MA_LENGTH if USE_VOLUME_FILTER else 0) + 2
        df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = calculate_atr(df)
        if USE_MA_FILTER:
            df = calculate_ma(df)
        if USE_RSI_FILTER:
            df = calculate_rsi(df)
        if USE_VOLUME_FILTER:
            df = calculate_volume_ma(df)

        close_prev = df['close'].iloc[-2]
        close = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]
        low = df['low'].iloc[-1]
        high = df['high'].iloc[-1]
        atr = df['atr'].iloc[-1]
        ma = df['ma'].iloc[-1] if USE_MA_FILTER else 0
        rsi = df['rsi'].iloc[-1] if USE_RSI_FILTER else 50
        volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1] if USE_VOLUME_FILTER else 0
        is_bearish = close < open_price
        enter_cond = close > close_prev
        if USE_MA_FILTER:
            enter_cond = enter_cond and (close > ma)
        if USE_RSI_FILTER:
            enter_cond = enter_cond and (rsi > RSI_THRESHOLD)
        if USE_VOLUME_FILTER:
            enter_cond = enter_cond and (volume > volume_ma)

        # Trailing TP (chequea current price)
        if in_position and USE_TRAILING_TP:
            max_price = max(max_price, high)
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            if current_price <= tp_price:
                order = exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"TP hit: Vendido {position_size} a {current_price}")
                in_position = False
                position_size = 0.0

        # SL update y chequeo
        if in_position and USE_ATR_SL:
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl)
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            if current_price <= sl_price:
                order = exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"SL hit: Vendido {position_size} a {current_price}")
                in_position = False
                position_size = 0.0

        # Salida bearish
        if in_position and is_bearish:
            order = exchange.create_market_sell_order(SYMBOL, position_size)
            print(f"Vela bajista: Vendido {position_size} a {close}")
            in_position = False
            position_size = 0.0

        # Entrada
        can_enter = enter_cond and (not in_position or ALLOW_REENTRY)
        if can_enter:
            balance = exchange.fetch_balance()['EUR']['free']  # Cambiado a EUR
            risk_balance = balance * RISK_PERCENT
            qty = risk_balance / close * (1 - COMMISSION)
            if qty > 0:
                order = exchange.create_market_buy_order(SYMBOL, qty)
                position_size += qty
                sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                max_price = close
                print(f"Comprado {qty} a {close}")
                in_position = True

        next_bar_time = (df['timestamp'].iloc[-1] + pd.Timedelta(seconds=exchange.parse_timeframe(TIMEFRAME))).timestamp()
        sleep_time = max(0, next_bar_time - time.time())
        time.sleep(sleep_time)

if __name__ == "__main__":
    if MODE == 'simulate':
        start_timestamp = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_timestamp = datetime.strptime(END_DATE, '%Y-%m-%d') if END_DATE else datetime.now()
        days = (end_timestamp - start_timestamp).days
        limit = days + max(ATR_LENGTH, MA_LENGTH if USE_MA_FILTER else 0, RSI_LENGTH if USE_RSI_FILTER else 0, VOLUME_MA_LENGTH if USE_VOLUME_FILTER else 0) + 10
        df = fetch_ohlcv(SYMBOL, TIMEFRAME, since=START_DATE, limit=limit)
        trades, final_equity = simulate_strategy(df, INITIAL_CAPITAL)
        print("Simulación completada.")
    elif MODE == 'live':
        live_strategy()
    else:
        print("Modo inválido. Usa 'simulate' o 'live'.")