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
    def __init__(self, mode='paper'):
        """
        Inicializa el bot de trading.
        
        Args:
            mode: 'paper' para simular, 'live' para real
        """
        # Cargar configuración desde variables de entorno
        self.API_KEY = os.getenv('BINANCE_API_KEY')
        self.API_SECRET = os.getenv('BINANCE_API_SECRET')
        
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
        # Permitir override desde variables de entorno
        self.MODE = os.getenv('TRADING_MODE', mode).lower()
        if self.MODE not in ['paper', 'live']:
            raise ValueError("MODE debe ser 'paper' o 'live'")
            
        self.TIMEFRAME = '4h'  # Optimizado según backtesting
        
        # Pares optimizados (Top 4 del backtesting en 4h)
        self.SYMBOLS = [
            'ETH/USDT',   # +91.4% ROI
            'XRP/USDT',   # +86.9% ROI
            'BNB/USDT',   # +82.4% ROI
            'SOL/USDT',   # +75.6% ROI
        ]
        
        # Capital por par (cargar desde .env o usar default)
        try:
            self.CAPITAL_PER_PAIR = float(os.getenv('CAPITAL_PER_PAIR', '50.0'))
        except ValueError:
            self.CAPITAL_PER_PAIR = 50.0
            
        self.TOTAL_CAPITAL = self.CAPITAL_PER_PAIR * len(self.SYMBOLS)
        
        # Parámetros de Riesgo (v1.0 probados)
        self.COMMISSION = 0.001
        self.RISK_PERCENT = 0.02
        self.MIN_EQUITY = 10.0
        self.MAX_TRADES_PER_DAY = 2
        
        # Indicadores (v1.0)
        self.ATR_LENGTH = 14
        self.ATR_MULTIPLIER = 3.5
        self.MA_LENGTH = 50
        self.LONG_MA_LENGTH = 200
        self.ADX_LENGTH = 14
        self.ADX_THRESHOLD = 25
        self.TRAILING_TP_PERCENT = 0.60
        
        # Estado del bot
        self.positions = {}  # {symbol: {size, entry_price, sl_price, max_price}}
        self.equity = {}     # {symbol: current_equity}
        self.trades_log = []
        
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
            # get_data solo actualiza si es necesario (>4h), sino carga desde disco
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
        
        atr = [float('nan')] * len(df)
        for i in range(self.ATR_LENGTH, len(df)):
            if i == self.ATR_LENGTH:
                atr[i] = df['tr'].iloc[i-self.ATR_LENGTH+1:i+1].mean()
            else:
                atr[i] = (atr[i-1] * (self.ATR_LENGTH - 1) + df['tr'].iloc[i]) / self.ATR_LENGTH
        df['atr'] = atr
        df['atr'] = df['atr'].fillna(0)
        
        # Moving Averages
        df['ma'] = df['close'].rolling(window=self.MA_LENGTH).mean().fillna(0)
        df['long_ma'] = df['close'].rolling(window=self.LONG_MA_LENGTH).mean().fillna(0)
        
        # ADX
        df['dm_plus'] = (df['high'] - df['high'].shift(1)).where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
        df['dm_minus'] = (df['low'].shift(1) - df['low']).where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
        df['tr_sm'] = df['tr'].rolling(self.ADX_LENGTH).mean()
        df['di_plus'] = (df['dm_plus'].rolling(self.ADX_LENGTH).mean() / df['tr_sm']) * 100
        df['di_minus'] = (df['dm_minus'].rolling(self.ADX_LENGTH).mean() / df['tr_sm']) * 100
        df['dx'] = (abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
        df['adx'] = df['dx'].rolling(self.ADX_LENGTH).mean().fillna(0)
        
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
    
    def execute_buy(self, symbol, price, qty, sl_price, tp_price):
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
                self.telegram.notify_error(f"Error comprando {symbol}: {str(e)}")
                success = False
        
        # Notificar por Telegram
        if success and self.telegram.enabled:
            self.telegram.notify_buy(symbol, price, qty, cost, sl_price, tp_price)
        
        return success
    
    def execute_sell(self, symbol, price, qty, reason, entry_price, pnl):
        """Ejecuta orden de venta."""
        if self.MODE == 'paper':
            print(f"📉 PAPER SELL: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f} | Reason: {reason}")
            success = True
        else:
            try:
                order = self.exchange.create_market_sell_order(symbol, qty)
                print(f"✅ LIVE SELL: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f} | Reason: {reason}")
                success = True
            except Exception as e:
                print(f"❌ Error vendiendo {symbol}: {e}")
                self.telegram.notify_error(f"Error vendiendo {symbol}: {str(e)}")
                success = False
        
        # Notificar por Telegram
        if success and self.telegram.enabled:
            roi = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            self.telegram.notify_sell(symbol, price, qty, reason, pnl, roi)
        
        return success
    
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
                self.execute_sell(symbol, tp_price, position['size'], 'TP', position['entry_price'], pnl)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', tp_price, position['size'], 'TP', pnl)
                self.positions[symbol] = None
                return
            
            # Trailing SL MA
            if close < current['ma']:
                proceeds = position['size'] * close * (1 - self.COMMISSION)
                pnl = proceeds - (position['size'] * position['entry_price'])
                self.execute_sell(symbol, close, position['size'], 'MA_SL', position['entry_price'], pnl)
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
                self.execute_sell(symbol, position['sl_price'], position['size'], 'SL', position['entry_price'], pnl)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', position['sl_price'], position['size'], 'SL', pnl)
                self.positions[symbol] = None
                return
            
            # Bearish exit
            if close < open_price and close < current['long_ma']:
                proceeds = position['size'] * close * (1 - self.COMMISSION)
                pnl = proceeds - (position['size'] * position['entry_price'])
                self.execute_sell(symbol, close, position['size'], 'bearish', position['entry_price'], pnl)
                self.equity[symbol] += proceeds
                self.log_trade(symbol, 'sell', close, position['size'], 'bearish', pnl)
                self.positions[symbol] = None
                return
        
        # Gestión de entradas
        if position is None:
            if self.check_entry_signal(df):
                close = current['close']
                atr = current['atr']
                
                qty = self.calculate_position_size(symbol, close, atr)
                if qty > 0:
                    cost = qty * close * (1 + self.COMMISSION)
                    if cost <= self.equity[symbol]:
                        # Calcular SL y TP para la notificación
                        sl_price = close - atr * self.ATR_MULTIPLIER
                        tp_price = close * (1 + self.TRAILING_TP_PERCENT)  # TP estimado inicial
                        
                        if self.execute_buy(symbol, close, qty, sl_price, tp_price):
                            self.equity[symbol] -= cost
                            self.positions[symbol] = {
                                'size': qty,
                                'entry_price': close,
                                'sl_price': sl_price,
                                'max_price': current['high']
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
        print(f"{'='*70}\n")
        
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
