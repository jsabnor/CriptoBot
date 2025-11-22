import os
import time
import math
import pandas as pd
import ccxt
from datetime import datetime, timedelta

# Configuración inicial
API_KEY = 'lt2KQOoCSsxfHujtQNEnB9wDwCUJjZ1qOEhcUb3ws1aOamKTaOrjgzd74zyOZp2R'  # Reemplaza con tu API key de Binance
API_SECRET = 'iIVQXHN5PRXPxi64AjAR42Vz2BGFvp5eBh9LFQFHKkfjzeQef7dPjMmmEjxL5CJ9'  # Reemplaza con tu API secret
SYMBOL = 'BTC/USDT'  # DOGE/EUR
TIMEFRAME = '1d'  # Temporalidad
INITIAL_CAPITAL = 200.0  # Capital inicial en EUR
COMMISSION = 0.001  # Comisión por trade (0.1% = 0.001)
USE_ATR_SL = True  # Usar SL basado en ATR
ATR_LENGTH = 14
ATR_MULTIPLIER = 3.0  # Amplio
ALLOW_REENTRY = True  # Permitir reentrada
USE_MA_FILTER = True  # Filtro MA para entradas
MA_LENGTH = 50
USE_LONG_MA_EXIT = True  # No salir bearish si close > MA(200)
LONG_MA_LENGTH = 200
USE_TRAILING_SL_MA = True  # Salir si close < MA(50)
USE_ADX_FILTER = True  # Entra si ADX > 20
ADX_LENGTH = 14
ADX_THRESHOLD = 30  # Relajado
USE_RSI_FILTER = True  # Desactivado
RSI_LENGTH = 14
RSI_THRESHOLD = 60
USE_VOLUME_FILTER = False  # Desactivado
VOLUME_MA_LENGTH = 20
USE_TRAILING_TP = True  # Trailing take profit
TRAILING_TP_PERCENT = 0.70  # 70% para pumps
RISK_PERCENT = 0.1  # All-in

# Modo: 'live' para ejecución real, 'simulate' para backtest
MODE = 'simulate'  # Cambia a 'live' para trading real
START_DATE = '2017-01-01'  # Para simulación: fecha inicio (YYYY-MM-DD)
END_DATE = '2025-12-31'  # Para simulación: fecha fin (YYYY-MM-DD) o actual
LOG_TRADES = True  # Guardar trades en CSV

# Parámetros de paginación / rate-limit para fetch_ohlcv_range
FETCH_MAX_LIMIT = 1000               # Ajusta si el exchange admite más/menos (Binance suele 1000)
FETCH_SLEEP_BETWEEN = 0.5           # segundos entre requests
FETCH_MAX_RETRIES = 5                # reintentos por fallo temporal

# Conexión a Binance via ccxt
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},  # Cambia a 'future' si usas futuros
})


def fetch_ohlcv_range(symbol, timeframe, since=None, until=None, max_limit=FETCH_MAX_LIMIT, sleep_between_requests=FETCH_SLEEP_BETWEEN):
    """
    Descarga todas las velas entre since (inclusive) y until (exclusive) paginando.
    - since: 'YYYY-MM-DD' or None
    - until: 'YYYY-MM-DD' or None
    Retorna DataFrame con columnas timestamp, open, high, low, close, volume (timestamp datetime).
    """
    # Convertir fechas a timestamps ms
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
        # ccxt expects since in ms and limit as int
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
            # nothing returned but maybe there are no more candles
            break

        consecutive_empty = 0
        all_ohlcv.extend(chunk)

        last_ts = chunk[-1][0]
        # Avanzar el since al siguiente ms después de la última vela recibida
        fetch_since = last_ts + 1

        # Si until definido y ya hemos alcanzado o superado, rompemos
        if until_ts and fetch_since >= until_ts:
            break

        # Si la respuesta devuelve menos de max_limit, muy probable que no hay más datos
        if len(chunk) < max_limit:
            break

        time.sleep(sleep_between_requests)

    # Construir DataFrame y eliminar duplicados por solapamientos
    if not all_ohlcv:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Si until está definido, truncar las filas cuyo timestamp >= until_ts (since until exclusive)
    if until_ts:
        df = df[df['timestamp'] < pd.to_datetime(until_ts, unit='ms')].reset_index(drop=True)

    return df


def calculate_atr(df, length=ATR_LENGTH):
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
    df = df.copy()
    df['ma'] = df['close'].rolling(window=length).mean()
    df['ma'] = df['ma'].fillna(0)
    return df


def calculate_long_ma(df, length=LONG_MA_LENGTH):
    df = df.copy()
    df['long_ma'] = df['close'].rolling(window=length).mean()
    df['long_ma'] = df['long_ma'].fillna(0)
    return df


def calculate_adx(df, length=ADX_LENGTH):
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
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=length).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    return df


def calculate_volume_ma(df, length=VOLUME_MA_LENGTH):
    df = df.copy()
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
            is_bearish = False  # No salir si en tendencia larga

        # Trailing TP
        if in_position:
            max_price = max(max_price, high)
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            if USE_TRAILING_TP and low <= tp_price:
                sell_price = tp_price
                proceeds = position_size * sell_price * (1 - COMMISSION)
                equity += proceeds
                trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'TP', 'equity': equity})
                position_size = 0.0
                in_position = False
                continue

        # Trailing SL MA
        if in_position and USE_TRAILING_SL_MA and close < ma:
            sell_price = close
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'MA_SL', 'equity': equity})
            position_size = 0.0
            in_position = False
            continue

        # SL update
        if USE_ATR_SL and in_position:
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl)

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
            qty = equity / close * (1 - COMMISSION)
            if qty > 0:
                position_size += qty
                equity -= qty * close
                sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                max_price = high
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
        limit = max(ATR_LENGTH, MA_LENGTH if USE_MA_FILTER or USE_TRAILING_SL_MA else 0, LONG_MA_LENGTH if USE_LONG_MA_EXIT else 0, ADX_LENGTH if USE_ADX_FILTER else 0, RSI_LENGTH if USE_RSI_FILTER else 0, VOLUME_MA_LENGTH if USE_VOLUME_FILTER else 0) + 2
        # para live obtenemos las últimas 'limit' velas (paginado por si hace falta)
        df = fetch_ohlcv_range(SYMBOL, TIMEFRAME, since=None, until=None, max_limit=limit)
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

        if len(df) < 2:
            time.sleep(5)
            continue

        close_prev = df['close'].iloc[-2]
        close = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]
        low = df['low'].iloc[-1]
        high = df['high'].iloc[-1]
        atr = df['atr'].iloc[-1]
        ma = df['ma'].iloc[-1] if USE_MA_FILTER or USE_TRAILING_SL_MA else 0
        long_ma = df['long_ma'].iloc[-1] if USE_LONG_MA_EXIT else 0
        adx = df['adx'].iloc[-1] if USE_ADX_FILTER else 0
        rsi = df['rsi'].iloc[-1] if USE_RSI_FILTER else 50
        volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1] if USE_VOLUME_FILTER else 0
        is_bearish = close < open_price
        if USE_LONG_MA_EXIT and close > long_ma:
            is_bearish = False
        enter_cond = close > close_prev
        if USE_MA_FILTER:
            enter_cond = enter_cond and (close > ma)
        if USE_ADX_FILTER:
            enter_cond = enter_cond and (adx > ADX_THRESHOLD)
        if USE_RSI_FILTER:
            enter_cond = enter_cond and (rsi > RSI_THRESHOLD)
        if USE_VOLUME_FILTER:
            enter_cond = enter_cond and (volume > volume_ma)

        # Trailing TP
        if in_position and USE_TRAILING_TP:
            max_price = max(max_price, high)
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            if current_price <= tp_price:
                try:
                    order = exchange.create_market_sell_order(SYMBOL, position_size)
                    print(f"TP hit: Vendido {position_size} a {current_price}")
                except Exception as e:
                    print(f"Error al vender por TP: {e}")
                in_position = False
                position_size = 0.0

        # Trailing SL MA
        if in_position and USE_TRAILING_SL_MA and close < ma:
            try:
                order = exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"MA SL hit: Vendido {position_size} a {close}")
            except Exception as e:
                print(f"Error al vender por MA_SL: {e}")
            in_position = False
            position_size = 0.0

        # SL update y chequeo
        if in_position and USE_ATR_SL:
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl)
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            if current_price <= sl_price:
                try:
                    order = exchange.create_market_sell_order(SYMBOL, position_size)
                    print(f"SL hit: Vendido {position_size} a {current_price}")
                except Exception as e:
                    print(f"Error al vender por SL: {e}")
                in_position = False
                position_size = 0.0

        # Salida bearish
        if in_position and is_bearish:
            try:
                order = exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"Vela bajista: Vendido {position_size} a {close}")
            except Exception as e:
                print(f"Error al vender por Bearish: {e}")
            in_position = False
            position_size = 0.0

        # Entrada
        can_enter = enter_cond and (not in_position or ALLOW_REENTRY)
        if can_enter:
            balance = exchange.fetch_balance().get('EUR', {}).get('free', 0.0)
            qty = balance / close * (1 - COMMISSION)
            if qty > 0:
                try:
                    order = exchange.create_market_buy_order(SYMBOL, qty)
                    print(f"Comprado {qty} a {close}")
                    position_size += qty
                    sl_price = close - atr * ATR_MULTIPLIER if USE_ATR_SL else 0
                    max_price = close
                    in_position = True
                except Exception as e:
                    print(f"Error al comprar: {e}")

        # Esperar hasta la siguiente barra (alineado aproximadamente)
        next_bar_time = (df['timestamp'].iloc[-1] + pd.Timedelta(seconds=exchange.parse_timeframe(TIMEFRAME))).timestamp()
        sleep_time = max(0, next_bar_time - time.time())
        time.sleep(min(sleep_time, 60))


if __name__ == "__main__":
    if MODE == 'simulate':
        start_timestamp = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_timestamp = datetime.strptime(END_DATE, '%Y-%m-%d') if END_DATE else datetime.now()
        # Obtener todo el rango con paginación
        df = fetch_ohlcv_range(SYMBOL, TIMEFRAME, since=START_DATE, until=END_DATE, max_limit=FETCH_MAX_LIMIT, sleep_between_requests=FETCH_SLEEP_BETWEEN)
        # Guardar CSV bruto para iteraciones rápidas si se desea
        os.makedirs('data', exist_ok=True)
        csv_path = f"data/{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{START_DATE}_{END_DATE}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Velas guardadas en {csv_path}, {len(df)} filas")

        trades, final_equity = simulate_strategy(df, INITIAL_CAPITAL)
        print("Simulación completada.")
    elif MODE == 'live':
        live_strategy()
    else:
        print("Modo inválido. Usa 'simulate' o 'live'.")
