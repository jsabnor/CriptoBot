import os
import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import json
from dotenv import load_dotenv
from telegram_notifier import TelegramNotifier
from data_cache import DataCache

# Cargar variables de entorno desde archivo .env
load_dotenv()

import config

# ============================================================================
# BOT v1.0 PRODUCTION - OPTIMIZADO PARA 4H
# ============================================================================
# Configuración basada en backtesting multi-activo:
# - Timeframe: 4h (mejor rendimiento)
# - Pares: ETH, XRP, BNB, SOL (top performers)
# - ROI esperado: +60-90% anual
# - Frecuencia: ~60 trades/año por par
# ============================================================================

class TradingBot:
    def __init__(self, mode=None):
        """
        Inicializa el bot de trading.
        """
        # Cargar credenciales desde config
        self.API_KEY = config.API_KEY
        self.API_SECRET = config.API_SECRET
        
        # Validar que las claves existan
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError(
                "ERROR: Claves API no configuradas.\n"
                "Crea un archivo .env con:\n"
                "BINANCE_API_KEY=tu_clave\n"
                "BINANCE_API_SECRET=tu_secreto\n"
                "Ver .env.example para más detalles."
            )
        
        # Configuración de Trading
        self.MODE = mode if mode else config.TRADING_MODE
        if self.MODE not in ['paper', 'live']:
            raise ValueError("MODE debe ser 'paper' o 'live'")
            
        self.TIMEFRAME = config.TIMEFRAME
        self.SYMBOLS = config.SYMBOLS
        self.CAPITAL_PER_PAIR = config.CAPITAL_PER_PAIR
        self.TOTAL_CAPITAL = self.CAPITAL_PER_PAIR * len(self.SYMBOLS)
        
        # Parámetros de Riesgo
        self.COMMISSION = config.COMMISSION
        self.RISK_PERCENT = config.RISK_PERCENT
        self.MIN_EQUITY = config.MIN_EQUITY
        self.MAX_TRADES_PER_DAY = config.MAX_TRADES_PER_DAY
        
        # Indicadores
        self.ATR_LENGTH = config.ATR_LENGTH
        self.ATR_MULTIPLIER = config.ATR_MULTIPLIER
        self.MA_LENGTH = config.MA_LENGTH
        self.LONG_MA_LENGTH = config.LONG_MA_LENGTH
        self.ADX_LENGTH = config.ADX_LENGTH
        self.ADX_THRESHOLD = config.ADX_THRESHOLD
        self.TRAILING_TP_PERCENT = config.TRAILING_TP_PERCENT
        
        # Estado del bot
        self.positions = {}  # {symbol: {size, entry_price, sl_price, max_price}}
        self.equity = {}     # {symbol: current_equity}
        self.trades_log = []
        self.last_summary_date = None  # Track last daily summary date
        
        # Inicializar equity por par
        for symbol in self.SYMBOLS:
            self.equity[symbol] = self.CAPITAL_PER_PAIR
            self.positions[symbol] = None
        
        # Exchange
        self.exchange = ccxt.binance({
            'apiKey': self.API_KEY,
            'secret': self.API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        
        # Telegram notifier
        self.telegram = TelegramNotifier()
        
        # Data cache para históricos
        self.data_cache = DataCache()
        
        # Imprimir configuración
        print(f"\n{'='*70}")
        print(f"BOT v1.0 PRODUCTION - MODO: {self.MODE.upper()}")
        print(f"{'='*70}")
        print(f"Timeframe: {self.TIMEFRAME}")
        print(f"Pares: {len(self.SYMBOLS)}")
        for i, symbol in enumerate(self.SYMBOLS, 1):
            print(f"  {i}. {symbol} (Capital: {self.CAPITAL_PER_PAIR} EUR)")
        print(f"Capital Total: {self.TOTAL_CAPITAL} EUR")
        print(f"Telegram: {'✓ Habilitado' if self.telegram.enabled else '✗ Deshabilitado'}")
        print(f"Caché de datos: ✓ Activo")
        print(f"{'='*70}\n")
        
        # NUEVO: Precargar caché de datos al inicio
        print("📊 Inicializando caché de datos históricos...")
        for symbol in self.SYMBOLS:
            # get_data actualiza si han pasado >5min, sino carga desde disco
            self.data_cache.get_data(symbol, self.TIMEFRAME)
        print("✅ Caché inicializado correctamente\n")
        
        # Notificar inicio por Telegram
        if self.telegram.enabled:
            self.telegram.notify_startup(self.MODE, self.SYMBOLS, self.TOTAL_CAPITAL)
    
    def fetch_ohlcv(self, symbol, limit=None):
        """Descarga datos OHLCV desde caché (con fallback a API)."""
        try:
            # Usar caché primero
            df = self.data_cache.get_data(symbol, self.TIMEFRAME)
            
            if df is None or len(df) < self.LONG_MA_LENGTH + 1:
                print(f"⚠️ Caché insuficiente para {symbol}, usando API...")
                return self._fetch_ohlcv_api(symbol, limit or 500)
            
            # Opcional: limitar a últimas N velas
            if limit:
                df = df.tail(limit)
            
            return df
            
        except Exception as e:
            print(f"❌ Error en caché {symbol}: {e}")
            return self._fetch_ohlcv_api(symbol, limit or 500)
    
    def _fetch_ohlcv_api(self, symbol, limit):
        """Fallback: fetch directo desde Binance API."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error API {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calcula indicadores técnicos."""
        df = df.copy()
        
        # ATR
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['prev_close']).abs(),
            (df['low'] - df['prev_close']).abs()
        ], axis=1).max(axis=1)
        
        # ATR (Wilder's Smoothing)
        alpha = 1 / self.ATR_LENGTH
        df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean().fillna(0)
        
        # Moving Averages
        df['ma'] = df['close'].rolling(window=self.MA_LENGTH).mean().fillna(0)
        df['long_ma'] = df['close'].rolling(window=self.LONG_MA_LENGTH).mean().fillna(0)
        
        # ADX (Standard Welles Wilder - Matching TradingView)
        df['dm_plus'] = (df['high'] - df['high'].shift(1)).where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
        df['dm_minus'] = (df['low'].shift(1) - df['low']).where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
        
        # Smoothing alpha for ADX
        alpha_adx = 1 / self.ADX_LENGTH
        
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
    
    def check_entry_signal(self, df):
        """Verifica señal de entrada (v1.0)."""
        if len(df) < self.LONG_MA_LENGTH + 1:
            return False
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Condiciones v1.0
        bullish = current['close'] > prev['close']
        above_ma = current['close'] > current['ma']
        strong_trend = current['adx'] > self.ADX_THRESHOLD
        long_term_bull = current['close'] > current['long_ma'] if current['long_ma'] > 0 else True
        
        return bullish and above_ma and strong_trend and long_term_bull
    
    def calculate_position_size(self, symbol, close, atr):
        """Calcula tamaño de posición basado en riesgo."""
        equity = self.equity[symbol]
        
        if equity < self.MIN_EQUITY or atr == 0:
            return 0
        
        risk_amount = equity * self.RISK_PERCENT
        stop_distance = atr * self.ATR_MULTIPLIER
        qty = (risk_amount / stop_distance) * (1 - self.COMMISSION)
        
        # Limitar a máximo 20% del equity
        max_qty = (equity * 0.20) / close
        qty = min(qty, max_qty)
        
        # Mínimo $1 por trade
        if qty * close < 1.0:
            return 0
        
        return qty
    
    def execute_buy(self, symbol, price, qty, sl_price, tp_price, adx=None, ma_status=None):
        """Ejecuta orden de compra."""
        cost = qty * price * (1 + self.COMMISSION)
        
        if self.MODE == 'paper':
            print(f"📈 PAPER BUY: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f}")
            success = True
        else:
            try:
                order = self.exchange.create_market_buy_order(symbol, qty)
                print(f"✅ LIVE BUY: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f}")
                success = True
            except Exception as e:
                print(f"❌ Error comprando {symbol}: {e}")
                if self.telegram.enabled:
                    self.telegram.notify_error(f"Error comprando {symbol}: {str(e)}")
                success = False
        
        # Notificar por Telegram
        if success and self.telegram.enabled:
            self.telegram.notify_buy(symbol, price, qty, cost, sl_price, tp_price, adx, ma_status, strategy_name='ADX')
        
        return success
    
    def execute_sell(self, symbol, price, qty, reason, entry_price, pnl, duration=None):
        """Ejecuta orden de venta."""
        if self.MODE == 'paper':
            print(f"📉 PAPER SELL: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f} | PnL: ${pnl:.2f}")
        else:
            try:
                order = self.exchange.create_market_sell_order(symbol, qty)
                print(f"✅ LIVE SELL: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f}")
            except Exception as e:
                print(f"❌ Error vendiendo {symbol}: {e}")
                if self.telegram.enabled:
                    self.telegram.notify_error(f"Error vendiendo {symbol}: {str(e)}")
                return
        
        # Notificar por Telegram
        if self.telegram.enabled:
            roi = (pnl / (entry_price * qty)) * 100
            self.telegram.notify_sell(symbol, price, qty, reason, pnl, roi, entry_price, duration, strategy_name='ADX')
    
    def calculate_trade_duration(self, entry_time):
        """Calcula la duración del trade en formato legible"""
        if not entry_time:
            return None
        
        duration = datetime.now() - entry_time
        hours = duration.total_seconds() / 3600
        
        if hours < 1:
            minutes = int(duration.total_seconds() / 60)
            return f"{minutes}m"
        elif hours < 24:
            return f"{int(hours)}h {int((hours % 1) * 60)}m"
        else:
            days = int(hours / 24)
            remaining_hours = int(hours % 24)
            return f"{days}d {remaining_hours}h"
    
    def process_symbol(self, symbol):
        """Procesa un símbolo (chequea entradas/salidas)."""
        # Descargar datos
        df = self.fetch_ohlcv(symbol)
        if df is None or len(df) < self.LONG_MA_LENGTH + 1:
            return
        
        # Calcular indicadores
        df = self.calculate_indicators(df)
        current = df.iloc[-1]
        
        position = self.positions[symbol]
        
        # Gestión de salidas
        if position is not None:
            close = current['close']
            low = current['low']
            high = current['high']
            open_price = current['open']
            
            # Actualizar max price
            position['max_price'] = max(position['max_price'], high)
            
            # Trailing TP
            tp_price = position['max_price'] * (1 - self.TRAILING_TP_PERCENT)
            if low <= tp_price:
                proceeds = position['size'] * tp_price * (1 - self.COMMISSION)
                pnl = proceeds - (position['size'] * position['entry_price'])
                duration = self.calculate_trade_duration(position.get('entry_time'))
                self.execute_sell(symbol, tp_price, position['size'], 'TP', position['entry_price'], pnl, duration)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', tp_price, position['size'], 'TP', pnl)
                self.positions[symbol] = None
                return
            
            # Trailing SL MA
            if close < current['ma']:
                proceeds = position['size'] * close * (1 - self.COMMISSION)
                pnl = proceeds - (position['size'] * position['entry_price'])
                duration = self.calculate_trade_duration(position.get('entry_time'))
                self.execute_sell(symbol, close, position['size'], 'MA_SL', position['entry_price'], pnl, duration)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', close, position['size'], 'MA_SL', pnl)
                self.positions[symbol] = None
                return
            
            # SL ATR
            new_sl = close - current['atr'] * self.ATR_MULTIPLIER
            position['sl_price'] = max(position['sl_price'], new_sl)
            if low <= position['sl_price']:
                proceeds = position['size'] * position['sl_price'] * (1 - self.COMMISSION)
                pnl = proceeds - (position['size'] * position['entry_price'])
                duration = self.calculate_trade_duration(position.get('entry_time'))
                self.execute_sell(symbol, position['sl_price'], position['size'], 'SL', position['entry_price'], pnl, duration)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', position['sl_price'], position['size'], 'SL', pnl)
                self.positions[symbol] = None
                return
            
            # Bearish exit
            if close < open_price and close < current['long_ma']:
                proceeds = position['size'] * close * (1 - self.COMMISSION)
                pnl = proceeds - (position['size'] * position['entry_price'])
                duration = self.calculate_trade_duration(position.get('entry_time'))
                self.execute_sell(symbol, close, position['size'], 'bearish', position['entry_price'], pnl, duration)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', close, position['size'], 'bearish', pnl)
                self.positions[symbol] = None
                return
        
        # Gestión de entradas
        if position is None:
            if self.check_entry_signal(df):
                close = current['close']
                atr = current['atr']
                adx = current['adx']
                ma = current['ma']
                
                # Determinar estado de MA
                ma_status = 'bullish' if close > ma else 'bearish'
                
                qty = self.calculate_position_size(symbol, close, atr)
                if qty > 0:
                    cost = qty * close * (1 + self.COMMISSION)
                    if cost <= self.equity[symbol]:
                        # Calcular SL y TP para la notificación
                        sl_price = close - atr * self.ATR_MULTIPLIER
                        tp_price = close * (1 + self.TRAILING_TP_PERCENT)  # TP estimado inicial
                        
                        if self.execute_buy(symbol, close, qty, sl_price, tp_price, adx, ma_status):
                            self.equity[symbol] -= cost
                            self.positions[symbol] = {
                                'size': qty,
                                'entry_price': close,
                                'sl_price': sl_price,
                                'max_price': current['high'],
                                'entry_time': datetime.now()  # Añadir timestamp de entrada
                            }
                            self.log_trade(symbol, 'buy', close, qty, '', 0)
    
    def log_trade(self, symbol, type, price, qty, reason, pnl):
        """Registra un trade."""
        self.trades_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'type': type,
            'price': price,
            'qty': qty,
            'reason': reason,
            'pnl': pnl,
            'equity': self.equity[symbol]
        })
    
    def save_state(self):
        """Guarda el estado actual."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.MODE,
            'equity': self.equity,
            'positions': {
                symbol: {
                    'size': pos['size'],
                    'entry_price': pos['entry_price'],
                    'sl_price': pos['sl_price'],
                    'max_price': pos['max_price']
                } if pos else None
                for symbol, pos in self.positions.items()
            },
            'total_equity': sum(self.equity.values()) + sum(
                pos['size'] * self.exchange.fetch_ticker(symbol)['last']
                if pos and self.MODE == 'live' else 0
                for symbol, pos in self.positions.items()
            )
        }
        
        with open('bot_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        if len(self.trades_log) > 0:
            df = pd.DataFrame(self.trades_log)
            df.to_csv('trades_production.csv', index=False)
    
    def run_once(self):
        """Ejecuta un ciclo de trading (procesa todos los pares)."""
        print(f"\n{'='*70}")
        print(f"🔄 CICLO DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        for symbol in self.SYMBOLS:
            print(f"\n📊 Procesando {symbol}...")
            self.process_symbol(symbol)
        
        # Mostrar estado
        print(f"\n{'='*70}")
        print(f"ESTADO ACTUAL")
        print(f"{'='*70}")
        
        total_equity = 0
        for symbol in self.SYMBOLS:
            equity = self.equity[symbol]
            position = self.positions[symbol]
            
            if position:
                # Si hay posición, sumar valor actual
                try:
                    current_price = self.exchange.fetch_ticker(symbol)['last'] if self.MODE == 'live' else position['entry_price']
                    position_value = position['size'] * current_price
                    total = equity + position_value
                except:
                    total = equity
            else:
                total = equity
            
            total_equity += total
            status = f"IN ({position['size']:.6f} @ ${position['entry_price']:.2f})" if position else "OUT"
            print(f"{symbol:>12}: ${total:>8.2f} | Status: {status}")
        
        print(f"{'='*70}")
        print(f"TOTAL EQUITY: ${total_equity:.2f} (ROI: {((total_equity - self.TOTAL_CAPITAL) / self.TOTAL_CAPITAL * 100):.2f}%)")
        print(f"Total Trades: {len([t for t in self.trades_log if t['type'] == 'buy'])}")
        print(f"Total Trades: {len([t for t in self.trades_log if t['type'] == 'buy'])}")
        print(f"{'='*70}\n")
        
    def send_daily_summary(self):
        """Envía resumen diario por Telegram"""
        if not self.telegram.enabled:
            return
            
        try:
            # Calcular estadísticas del día
            today_pnl = 0
            today_roi = 0
            today_trades = 0
            today_wins = 0
            today_losses = 0
            
            if os.path.exists('trades_production.csv'):
                df = pd.read_csv('trades_production.csv')
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Filtrar últimas 24h
                    now = datetime.now()
                    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                    today_mask = df['timestamp'] >= start_of_day
                    df_today = df[today_mask]
                    
                    if not df_today.empty:
                        # En bot ADX, los trades cerrados son 'sell'
                        sells = df_today[df_today['type'] == 'sell']
                        today_trades = len(sells)
                        if today_trades > 0:
                            today_pnl = sells['pnl'].sum()
                            today_wins = len(sells[sells['pnl'] > 0])
                            today_losses = len(sells[sells['pnl'] <= 0])
                            
                            # ROI del día basado en capital inicial
                            today_roi = (today_pnl / self.TOTAL_CAPITAL) * 100
            
            # Totales
            total_equity = 0
            for symbol in self.SYMBOLS:
                equity = self.equity[symbol]
                position = self.positions[symbol]
                if position:
                    try:
                        current_price = self.exchange.fetch_ticker(symbol)['last'] if self.MODE == 'live' else position['entry_price']
                        position_value = position['size'] * current_price
                        total_equity += equity + position_value
                    except:
                        total_equity += equity
                else:
                    total_equity += equity
            
            total_roi = ((total_equity - self.TOTAL_CAPITAL) / self.TOTAL_CAPITAL * 100)
            open_positions = sum(1 for p in self.positions.values() if p is not None)
            
            stats = {
                'pnl': today_pnl,
                'roi': today_roi,
                'total_trades': today_trades,
                'wins': today_wins,
                'losses': today_losses,
                'win_rate': (today_wins / today_trades * 100) if today_trades > 0 else 0,
                'total_equity': total_equity,
                'total_roi': total_roi,
                'open_positions': open_positions
            }
            
            self.telegram.notify_daily_summary(stats)
            print(f"✅ Resumen diario enviado: PnL ${today_pnl:.2f}")
            
        except Exception as e:
            print(f"❌ Error enviando resumen diario: {e}")
        
        self.save_state()
        
        # Notificar ciclo completado por Telegram
        if self.telegram.enabled:
            open_positions = sum(1 for p in self.positions.values() if p is not None)
            roi = ((total_equity - self.TOTAL_CAPITAL) / self.TOTAL_CAPITAL * 100)
            self.telegram.notify_cycle_complete(total_equity, self.TOTAL_CAPITAL, roi, open_positions)
    
    def run_continuous(self, interval_hours=4):
        """Ejecuta el bot continuamente, sincronizado con el cierre de velas."""
        from datetime import timedelta
        
        print(f"\n🚀 Iniciando bot en modo continuo...")
        print(f"⏰ Timeframe: {interval_hours}h (sincronizado con cierre de velas)")
        print(f"🛑 Presiona Ctrl+C para detener")
        print(f"{'='*70}\n")
        
        while True:
            try:
                # Calcular próximo cierre de vela
                now = datetime.utcnow()
                
                # Velas de 4h cierran a las 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
                current_hour = now.hour
                next_close_hour = ((current_hour // interval_hours) + 1) * interval_hours
                
                if next_close_hour >= 24:
                    # Próximo cierre es mañana a las 00:00
                    next_close = datetime(now.year, now.month, now.day) + timedelta(days=1)
                else:
                    # Próximo cierre es hoy
                    next_close = datetime(now.year, now.month, now.day, next_close_hour, 0, 0)
                
                # Añadir 30 segundos después del cierre para asegurar datos completos
                next_run = next_close + timedelta(seconds=30)
                
                # Calcular tiempo de espera
                wait_seconds = (next_run - now).total_seconds()
                
                if wait_seconds > 0:
                    wait_minutes = int(wait_seconds / 60)
                    print(f"⏳ Esperando al cierre de vela en {next_close.strftime('%H:%M:%S')} UTC")
                    print(f"   Próximo análisis: {next_run.strftime('%H:%M:%S')} UTC ({wait_minutes} min)")
                    time.sleep(wait_seconds)
                
                # Ejecutar ciclo de trading
                self.run_once()
                
                # Enviar resumen diario si cambió el día y aún no se envió hoy
                current_time = datetime.utcnow()
                current_date = current_time.date()
                
                # Enviar solo una vez al día, en el primer ciclo después de las 00:00 UTC
                if current_date != self.last_summary_date and current_time.hour >= 0:
                    self.send_daily_summary()
                    self.last_summary_date = current_date
                
            except KeyboardInterrupt:
                print("\n\n🛑 Bot detenido por el usuario")
                break
            except Exception as e:
                print(f"\n❌ Error en el ciclo continuo: {e}")
                if self.telegram.enabled:
                    self.telegram.notify_error(f"Error en ciclo continuo: {str(e)}")
                print("⏳ Reintentando en 5 minutos...")
                time.sleep(300)


if __name__ == "__main__":
    # Cargar modo desde variable de entorno, por defecto 'paper'
    MODE = os.getenv('TRADING_MODE', 'paper').lower()
    
    # Validar modo
    if MODE not in ['paper', 'live']:
        print(f"ERROR: TRADING_MODE debe ser 'paper' o 'live', recibido: '{MODE}'")
        print("Usando modo 'paper' por seguridad")
        MODE = 'paper'
    
    # Advertencia si es modo live
    if MODE == 'live':
        print("\n" + "="*70)
        print("⚠️  ATENCIÓN: MODO LIVE ACTIVADO - USARÁS DINERO REAL ⚠️")
        print("="*70)
        print("Presiona Ctrl+C en los próximos 5 segundos para cancelar...\n")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n\nBot cancelado por el usuario")
            exit(0)
    
    bot = TradingBot(mode=MODE)
    
    # Ejecutar un solo ciclo (para testing)
    # bot.run_once()
    
    # O ejecutar continuamente
    bot.run_continuous(interval_hours=4)
