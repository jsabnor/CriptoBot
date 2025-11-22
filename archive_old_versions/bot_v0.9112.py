import os
import time
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime

# =========================
# Configuración
# =========================
API_KEY = 'lt2KQOoCSsxfHujtQNEnB9wDwCUJjZ1qOEhcUb3ws1aOamKTaOrjgzd74zyOZp2R'  # Reemplaza con tu API key de Binance
API_SECRET = 'iIVQXHN5PRXPxi64AjAR42Vz2BGFvp5eBh9LFQFHKkfjzeQef7dPjMmmEjxL5CJ9'  # Reemplaza con tu API secret
SYMBOL = 'BTC/USDT'  # DOGE/EUR
TIMEFRAME = '1d'

INITIAL_CAPITAL = 200.0                 # Capital inicial (USDT si operas BTC/USDT)
COMMISSION = 0.001                      # 0.1% por lado
ATR_LENGTH = 14
ATR_MULTIPLIER = 3.0                    # Stop basado en ATR (más realista que 5.0)
RISK_PERCENT = 0.10                     # 10% del capital por entrada (simulación simple tipo “fixed fraction”)

USE_MA_FILTER = True
MA_LENGTH = 50
USE_LONG_MA_EXIT = True                 # Si close > MA200 no aplicar salida bajista
LONG_MA_LENGTH = 200
USE_TRAILING_SL_MA = True               # Salida si close < MA50
USE_ADX_FILTER = True
ADX_LENGTH = 14
ADX_THRESHOLD = 20
USE_TRAILING_TP = True
TRAILING_TP_PERCENT = 0.30              # 30% de retroceso desde el máximo

MODE = 'simulate'                       # 'simulate' o 'live'
START_DATE = '2017-01-01'
END_DATE = '2025-12-31'
LOG_TRADES = True

FETCH_LIMIT = 1000
FETCH_SLEEP = 0.25
FETCH_RETRIES = 5

# =========================
# Conexión exchange (ccxt)
# =========================
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})


# =========================
# Descarga de datos
# =========================
def fetch_ohlcv_range(symbol, timeframe, since=None, until=None, limit=FETCH_LIMIT, sleep_between_requests=FETCH_SLEEP):
    if since:
        fetch_since = exchange.parse8601(since + 'T00:00:00Z')
    else:
        fetch_since = None

    until_ts = None
    if until:
        until_dt = datetime.strptime(until, '%Y-%m-%d')
        until_ts = int(until_dt.timestamp() * 1000)

    all_ohlcv = []
    while True:
        tries = 0
        while tries <= FETCH_RETRIES:
            try:
                chunk = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=limit)
                break
            except ccxt.BaseError as e:
                tries += 1
                wait = min(5 * tries, 60)
                print(f"[fetch_ohlcv] error: {e}. retry {tries}/{FETCH_RETRIES} after {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("fetch_ohlcv repeated failures")

        if not chunk:
            break

        all_ohlcv.extend(chunk)
        last_ts = chunk[-1][0]
        fetch_since = last_ts + 1

        if until_ts and fetch_since >= until_ts:
            break
        if len(chunk) < limit:
            break

        time.sleep(sleep_between_requests)

    if not all_ohlcv:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    if until_ts:
        df = df[df['timestamp'] < pd.to_datetime(until_ts, unit='ms')].reset_index(drop=True)
    return df


# =========================
# Indicadores
# =========================
def add_atr(df, length=ATR_LENGTH):
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    # Corrección del FutureWarning
    df['atr'] = atr.bfill().fillna(0.0)
    return df

def add_ma(df, length):
    df[f'ma_{length}'] = df['close'].rolling(length).mean().fillna(0.0)
    return df

def add_adx(df, length=ADX_LENGTH):
    h, l, c = df['high'], df['low'], df['close']
    up_move = h.diff()
    down_move = -l.diff()
    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr_components = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    tr = tr_components.rolling(length).mean()

    di_plus = (dm_plus.rolling(length).mean() / tr) * 100.0
    di_minus = (dm_minus.rolling(length).mean() / tr) * 100.0
    dx = ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, pd.NA)) * 100.0
    adx = dx.rolling(length).mean()
    df['adx'] = adx.fillna(0.0)
    return df


# =========================
# Utilidades de métricas
# =========================
def compute_backtest_stats(trades_df, equity_df):
    # Número de buys/sells
    buys = trades_df[trades_df['type'] == 'buy'].copy()
    sells = trades_df[trades_df['type'] == 'sell'].copy()

    num_trades = len(buys)
    # Aproximación: cambios de equity en ventas para clasificar ganadoras/perdedoras
    sells_equity = sells['equity'].diff()
    num_wins = (sells_equity > 0).sum() if sells_equity is not None else 0
    num_losses = (sells_equity < 0).sum() if sells_equity is not None else 0
    win_rate = (num_wins / (num_wins + num_losses) * 100.0) if (num_wins + num_losses) > 0 else 0.0

    # Retorno total
    if equity_df.empty:
        initial_equity = INITIAL_CAPITAL
        final_equity = INITIAL_CAPITAL
    else:
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]

    total_return_pct = ((final_equity - initial_equity) / initial_equity * 100.0) if initial_equity != 0 else 0.0

    # Drawdown máximo
    equity_series = equity_df['equity'] if not equity_df.empty else pd.Series([INITIAL_CAPITAL])
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown_pct = (drawdown.min() * 100.0) if len(drawdown) > 0 else 0.0

    # Profit factor
    profits = sells_equity.dropna()
    profit_factor = np.nan
    if not profits.empty:
        total_gain = profits[profits > 0].sum()
        total_loss = profits[profits < 0].sum()
        profit_factor = (total_gain / abs(total_loss)) if total_loss != 0 else np.nan

    # Sharpe ratio (simplificado con retornos diarios de equity)
    returns = equity_series.pct_change().dropna()
    sharpe = ((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() and returns.std() > 0 else np.nan

    return {
        'num_trades': num_trades,
        'num_wins': int(num_wins),
        'num_losses': int(num_losses),
        'win_rate': win_rate,
        'initial_equity': float(initial_equity),
        'final_equity': float(final_equity),
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'profit_factor': float(profit_factor) if profit_factor == profit_factor else np.nan,
        'sharpe_ratio': float(sharpe) if sharpe == sharpe else np.nan
    }


# =========================
# Simulación de estrategia
# =========================
def simulate_strategy(df, initial_capital):
    df = df.copy()
    df = add_atr(df, ATR_LENGTH)
    if USE_MA_FILTER or USE_TRAILING_SL_MA:
        df = add_ma(df, MA_LENGTH)
    if USE_LONG_MA_EXIT:
        df = add_ma(df, LONG_MA_LENGTH)
    if USE_ADX_FILTER:
        df = add_adx(df, ADX_LENGTH)

    equity = initial_capital
    position_size = 0.0
    sl_price = None
    max_price = None
    in_position = False

    trades = []
    equity_curve = []

    for i in range(1, len(df)):
        close_prev = df['close'].iloc[i-1]
        close = df['close'].iloc[i]
        open_price = df['open'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        atr = df['atr'].iloc[i]
        ma = df[f'ma_{MA_LENGTH}'].iloc[i] if USE_MA_FILTER or USE_TRAILING_SL_MA else None
        long_ma = df[f'ma_{LONG_MA_LENGTH}'].iloc[i] if USE_LONG_MA_EXIT else None
        adx = df['adx'].iloc[i] if USE_ADX_FILTER else None
        timestamp = df['timestamp'].iloc[i]

        # Condiciones de entrada
        enter_cond = close > close_prev
        if USE_MA_FILTER and ma is not None:
            enter_cond = enter_cond and (close > ma)
        if USE_ADX_FILTER and adx is not None:
            enter_cond = enter_cond and (adx >= ADX_THRESHOLD)

        # Salida bajista
        is_bearish = close < open_price
        if USE_LONG_MA_EXIT and long_ma is not None and close > long_ma:
            is_bearish = False

        # Trailing TP (retroceso desde el máximo)
        if in_position and USE_TRAILING_TP:
            max_price = max(max_price, high) if max_price is not None else high
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)
            if low <= tp_price:
                sell_price = tp_price
                proceeds = position_size * sell_price * (1 - COMMISSION)
                equity += proceeds
                trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'TP', 'equity': equity})
                position_size = 0.0
                in_position = False
                sl_price = None
                max_price = None
                # marcar equity
                equity_curve.append({'timestamp': timestamp, 'equity': equity})
                continue

        # Trailing SL por MA
        if in_position and USE_TRAILING_SL_MA and ma is not None and close < ma:
            sell_price = close
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': f'MA{MA_LENGTH}_SL', 'equity': equity})
            position_size = 0.0
            in_position = False
            sl_price = None
            max_price = None
            equity_curve.append({'timestamp': timestamp, 'equity': equity})
            continue

        # Actualizar SL por ATR
        if in_position and atr > 0:
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl) if sl_price is not None else new_sl

        # Salida por vela bajista
        if in_position and is_bearish:
            sell_price = close
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'bearish', 'equity': equity})
            position_size = 0.0
            in_position = False
            sl_price = None
            max_price = None

        # Salida por SL
        if in_position and (sl_price is not None) and low <= sl_price:
            sell_price = sl_price
            proceeds = position_size * sell_price * (1 - COMMISSION)
            equity += proceeds
            trades.append({'timestamp': timestamp, 'type': 'sell', 'price': sell_price, 'qty': position_size, 'reason': 'SL', 'equity': equity})
            position_size = 0.0
            in_position = False
            sl_price = None
            max_price = None

        # Entrada (fixed fraction del equity disponible)
        if enter_cond and not in_position:
            # Usar un porcentaje del equity para comprar
            notional_to_use = equity * RISK_PERCENT
            if notional_to_use > 0:
                qty = (notional_to_use / close) * (1 - COMMISSION)
                cost = qty * close
                if qty > 0 and cost <= equity:
                    position_size = qty
                    equity -= cost
                    sl_price = close - atr * ATR_MULTIPLIER if atr > 0 else None
                    max_price = high
                    trades.append({'timestamp': timestamp, 'type': 'buy', 'price': close, 'qty': qty, 'equity': equity})
                    in_position = True

        # Marcar equity mark-to-market
        mark_to_market = equity + (position_size * close if in_position else 0.0)
        equity_curve.append({'timestamp': timestamp, 'equity': mark_to_market})

    # Cierre final si queda posición
    if in_position:
        final_close = df['close'].iloc[-1]
        proceeds = position_size * final_close * (1 - COMMISSION)
        equity += proceeds
        trades.append({'timestamp': df['timestamp'].iloc[-1], 'type': 'sell', 'price': final_close, 'qty': position_size, 'reason': 'end', 'equity': equity})

    # Logs
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    if LOG_TRADES:
        trades_df.to_csv('trades_log.csv', index=False)
        equity_df.to_csv('equity_curve.csv', index=False)

    # Prints de consola básicos
    print(f"Equity final: {equity:.2f} USDT")
    print(f"Número de trades (buys): {len(trades_df[trades_df['type']=='buy'])}")
    print("Simulación completada. Logs guardados en trades_log.csv y equity_curve.csv")

    # Estadísticas adicionales en consola
    stats = compute_backtest_stats(trades_df, equity_df)
    print("\n📊 Estadísticas del Backtest")
    print(f"- Número de trades: {stats['num_trades']}")
    print(f"- Ganadores: {stats['num_wins']}, Perdedoras: {stats['num_losses']}")
    print(f"- Win Rate: {stats['win_rate']:.2f}%")
    print(f"- Equity inicial: {stats['initial_equity']:.2f}, Final: {stats['final_equity']:.2f}")
    print(f"- Retorno total: {stats['total_return_pct']:.2f}%")
    print(f"- Máximo Drawdown: {stats['max_drawdown_pct']:.2f}%")
    pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] == stats['profit_factor'] else "n/a"
    sh_str = f"{stats['sharpe_ratio']:.2f}" if stats['sharpe_ratio'] == stats['sharpe_ratio'] else "n/a"
    print(f"- Profit Factor: {pf_str}")
    print(f"- Sharpe Ratio: {sh_str}")

    return trades_df, equity_df, equity


# =========================
# Live (opcional, spot BTC/USDT)
# =========================
def live_strategy():
    print("Iniciando modo live... Ctrl+C para detener.")
    in_position = False
    position_size = 0.0
    sl_price = None
    max_price = None

    while True:
        limit = max(ATR_LENGTH, MA_LENGTH if USE_MA_FILTER or USE_TRAILING_SL_MA else 0,
                    LONG_MA_LENGTH if USE_LONG_MA_EXIT else 0,
                    ADX_LENGTH if USE_ADX_FILTER else 0) + 2

        df = fetch_ohlcv_range(SYMBOL, TIMEFRAME, since=None, until=None, limit=limit)

        if df.empty or len(df) < 2:
            time.sleep(5)
            continue

        df = add_atr(df, ATR_LENGTH)
        if USE_MA_FILTER or USE_TRAILING_SL_MA: df = add_ma(df, MA_LENGTH)
        if USE_LONG_MA_EXIT: df = add_ma(df, LONG_MA_LENGTH)
        if USE_ADX_FILTER: df = add_adx(df, ADX_LENGTH)

        close_prev = df['close'].iloc[-2]
        close = df['close'].iloc[-1]
        open_price = df['open'].iloc[-1]
        low = df['low'].iloc[-1]
        high = df['high'].iloc[-1]
        atr = df['atr'].iloc[-1]
        ma = df.get(f'ma_{MA_LENGTH}', pd.Series(index=df.index)).iloc[-1] if USE_MA_FILTER or USE_TRAILING_SL_MA else None
        long_ma = df.get(f'ma_{LONG_MA_LENGTH}', pd.Series(index=df.index)).iloc[-1] if USE_LONG_MA_EXIT else None
        adx = df['adx'].iloc[-1] if USE_ADX_FILTER else None

        enter_cond = close > close_prev
        if USE_MA_FILTER and ma is not None: enter_cond = enter_cond and (close > ma)
        if USE_ADX_FILTER and adx is not None: enter_cond = enter_cond and (adx >= ADX_THRESHOLD)

        is_bearish = close < open_price
        if USE_LONG_MA_EXIT and long_ma is not None and close > long_ma:
            is_bearish = False

        # Actualizar SL y TP
        if in_position and atr > 0:
            new_sl = close - atr * ATR_MULTIPLIER
            sl_price = max(sl_price, new_sl) if sl_price is not None else new_sl
        if in_position and USE_TRAILING_TP:
            max_price = max(max_price, high) if max_price is not None else high
            tp_price = max_price * (1 - TRAILING_TP_PERCENT)

        # Salidas
        if in_position:
            last_price = exchange.fetch_ticker(SYMBOL)['last']
            if sl_price is not None and last_price <= sl_price:
                exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"SL hit @ {last_price}")
                in_position = False; position_size = 0.0; sl_price = None; max_price = None
            elif USE_TRAILING_SL_MA and ma is not None and close < ma:
                exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"MA{MA_LENGTH} SL @ {close}")
                in_position = False; position_size = 0.0; sl_price = None; max_price = None
            elif is_bearish:
                exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"Bearish exit @ {close}")
                in_position = False; position_size = 0.0; sl_price = None; max_price = None
            elif USE_TRAILING_TP and last_price <= tp_price:
                exchange.create_market_sell_order(SYMBOL, position_size)
                print(f"TP giveback exit @ {last_price}")
                in_position = False; position_size = 0.0; sl_price = None; max_price = None

        # Entrada (usar USDT disponible)
        if not in_position and enter_cond:
            balance = exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            notional_to_use = usdt_free * RISK_PERCENT
            qty = (notional_to_use / close) * (1 - COMMISSION)
            if qty > 0 and (qty * close) <= usdt_free:
                exchange.create_market_buy_order(SYMBOL, qty)
                print(f"Entry qty {qty} @ {close}")
                in_position = True
                position_size = qty
                sl_price = close - atr * ATR_MULTIPLIER if atr > 0 else None
                max_price = close

        # Dormir hasta próxima barra
        next_bar_sec = exchange.parse_timeframe(TIMEFRAME)
        next_bar_time = (df['timestamp'].iloc[-1] + pd.Timedelta(seconds=next_bar_sec)).timestamp()
        sleep_time = max(0, next_bar_time - time.time())
        time.sleep(min(sleep_time, 60))


# =========================
# Main
# =========================
if __name__ == "__main__":
    if MODE == 'simulate':
        df = fetch_ohlcv_range(SYMBOL, TIMEFRAME, since=START_DATE, until=END_DATE, limit=FETCH_LIMIT, sleep_between_requests=FETCH_SLEEP)
        os.makedirs('data', exist_ok=True)
        csv_path = f"data/{SYMBOL.replace('/', '_')}_{TIMEFRAME}_{START_DATE}_{END_DATE}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Velas guardadas en {csv_path}, {len(df)} filas")

        trades_df, equity_df, final_equity = simulate_strategy(df, INITIAL_CAPITAL)
        print(f"\nFinal equity (sim): {final_equity:.2f} USDT")
        print(f"Total trades ejecutados: {len(trades_df)}")

    elif MODE == 'live':
        live_strategy()
    else:
        print("Modo inválido. Usa 'simulate' o 'live'.")
