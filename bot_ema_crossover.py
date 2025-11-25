"""
Trading Bot - EMA Crossover Strategy (Optimized)

Estrategia optimizada basada en cruces de EMAs:
- EMA Rápida: 15 períodos
- EMA Lenta: 30 períodos
- Timeframe: 4h
- Gestión de riesgo: 2% por trade

Resultados del backtest (2020-2025):
- ROI: +426%
- Win Rate: 27.5%
- Drawdown: -24.2%
- Score: 4.85

Uso:
    python bot_ema_crossover.py
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from data_cache import DataCache
from telegram_notifier import TelegramNotifier
import json

load_dotenv()

class EMABot:
    """Bot de trading con estrategia EMA Crossover optimizada"""
    
    # Configuración de la estrategia
    EMA_FAST = 15
    EMA_SLOW = 30
    RISK_PERCENT = 0.02  # 2% de riesgo por trade
    COMMISSION = 0.001   # 0.1%
    
    def __init__(self, mode='paper'):
        """
        Inicializa el bot
        
        Args:
            mode: 'paper' o 'live'
        """
        self.MODE = mode
        
        # Validar claves API
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError(
                "ERROR: Claves API no configuradas.\n"
                "Crea un archivo .env con:\n"
                "BINANCE_API_KEY=tu_clave\n"
                "BINANCE_API_SECRET=tu_secreto\n"
                "Ver .env.example para más detalles."
            )
        
        # Inicializar exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Configuración
        self.symbols = os.getenv('SYMBOLS', 'ETH/USDT,XRP/USDT,BNB/USDT,SOL/USDT').split(',')
        self.timeframe = os.getenv('TIMEFRAME', '4h')
        self.capital_per_pair = float(os.getenv('CAPITAL_PER_PAIR', '50'))
        
        # Estado del bot
        self.equity = {symbol: self.capital_per_pair for symbol in self.symbols}
        self.positions = {symbol: None for symbol in self.symbols}
        
        # Servicios
        self.cache = DataCache()
        self.telegram = TelegramNotifier()
        
        # Cargar estado si existe
        self.load_state()
        
        print(f"\n{'='*60}")
        print(f"EMA CROSSOVER BOT - MODO: {self.MODE.upper()}")
        print(f"{'='*60}")
        print(f"Estrategia: EMA {self.EMA_FAST}/{self.EMA_SLOW}")
        print(f"Símbolos: {', '.join(self.symbols)}")
        print(f"Capital por par: ${self.capital_per_pair}")
        print(f"Riesgo por trade: {self.RISK_PERCENT*100}%")
        print(f"{'='*60}\n")
    
    def calculate_ema(self, df, period):
        """Calcula EMA"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, df, period=14):
        """Calcula ATR para stop loss"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_indicators(self, df):
        """Calcula todos los indicadores necesarios"""
        df = df.copy()
        df['ema_fast'] = self.calculate_ema(df, self.EMA_FAST)
        df['ema_slow'] = self.calculate_ema(df, self.EMA_SLOW)
        df['atr'] = self.calculate_atr(df, 14)
        return df
    
    def check_entry_signal(self, df):
        """
        Verifica señal de entrada (compra)
        
        Señal: EMA rápida cruza EMA lenta al alza
        """
        if len(df) < self.EMA_SLOW + 1:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Cruce alcista
        signal = (
            current['ema_fast'] > current['ema_slow'] and
            previous['ema_fast'] <= previous['ema_slow']
        )
        
        return signal
    
    def check_exit_signal(self, df):
        """
        Verifica señal de salida (venta)
        
        Señal: EMA rápida cruza EMA lenta a la baja
        """
        if len(df) < self.EMA_SLOW + 1:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Cruce bajista
        signal = (
            current['ema_fast'] < current['ema_slow'] and
            previous['ema_fast'] >= previous['ema_slow']
        )
        
        return signal
    
    def calculate_position_size(self, symbol, entry_price, sl_price):
        """
        Calcula tamaño de posición basado en riesgo
        
        Arriesga 2% del capital por trade
        """
        capital = self.equity[symbol]
        risk_amount = capital * self.RISK_PERCENT
        risk_per_unit = entry_price - sl_price
        
        if risk_per_unit <= 0:
            return 0
        
        qty = risk_amount / risk_per_unit
        
        # Verificar que no exceda el capital disponible
        cost = qty * entry_price * (1 + self.COMMISSION)
        if cost > capital:
            qty = (capital * 0.95) / entry_price  # Usar 95% del capital
        
        return qty
    
    def execute_buy(self, symbol, price, qty, sl_price):
        """Ejecuta orden de compra"""
        cost = qty * price * (1 + self.COMMISSION)
        
        if self.MODE == 'paper':
            print(f"📈 PAPER BUY: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f} | SL: ${sl_price:.2f}")
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
            # Calcular TP estimado (usaremos 2:1 R:R)
            risk = price - sl_price
            tp_price = price + (risk * 2)
            
            self.telegram.notify_buy(symbol, price, qty, cost, sl_price, tp_price, strategy_name='EMA')
        
        return success
    
    def execute_sell(self, symbol, price, qty, reason, entry_price, pnl, duration=None):
        """Ejecuta orden de venta"""
        if self.MODE == 'paper':
            print(f"📉 PAPER SELL: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f} | Reason: {reason}")
            success = True
        else:
            try:
                order = self.exchange.create_market_sell_order(symbol, qty)
                print(f"✅ LIVE SELL: {symbol} | Qty: {qty:.8f} | Price: ${price:.2f}")
                success = True
            except Exception as e:
                print(f"❌ Error vendiendo {symbol}: {e}")
                self.telegram.notify_error(f"Error vendiendo {symbol}: {str(e)}")
                success = False
        
        # Notificar por Telegram
        if success and self.telegram.enabled:
            roi = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            self.telegram.notify_sell(symbol, price, qty, reason, pnl, roi, entry_price, duration, strategy_name='EMA')
        
        return success
    
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
        """Procesa un símbolo (chequea entradas/salidas)"""
        # Obtener datos
        df = self.cache.get_data(symbol, self.timeframe)
        if df is None or len(df) < self.EMA_SLOW + 1:
            return
        
        # Calcular indicadores
        df = self.calculate_indicators(df)
        current = df.iloc[-1]
        
        position = self.positions[symbol]
        
        # Gestión de salidas
        if position is not None:
            close = current['close']
            low = current['low']
            
            # Check stop loss
            if low <= position['sl_price']:
                proceeds = position['qty'] * position['sl_price'] * (1 - self.COMMISSION)
                pnl = proceeds - (position['qty'] * position['entry_price'])
                duration = self.calculate_trade_duration(position.get('entry_time'))
                
                if self.execute_sell(symbol, position['sl_price'], position['qty'], 'SL', 
                                   position['entry_price'], pnl, duration):
                    self.equity[symbol] += proceeds
                    self.log_trade(symbol, 'sell', position['sl_price'], position['qty'], 'SL', pnl)
                    self.positions[symbol] = None
                    return
            
            # Check señal de salida
            if self.check_exit_signal(df):
                proceeds = position['qty'] * close * (1 - self.COMMISSION)
                pnl = proceeds - (position['qty'] * position['entry_price'])
                duration = self.calculate_trade_duration(position.get('entry_time'))
                
                if self.execute_sell(symbol, close, position['qty'], 'Signal', 
                                   position['entry_price'], pnl, duration):
                    self.equity[symbol] += proceeds
                    self.log_trade(symbol, 'sell', close, position['qty'], 'Signal', pnl)
                    self.positions[symbol] = None
                    return
        
        # Gestión de entradas
        if position is None:
            if self.check_entry_signal(df):
                close = current['close']
                atr = current['atr']
                
                # Stop loss: 2 ATR por debajo del precio
                sl_price = close - (atr * 2)
                
                # Calcular tamaño de posición
                qty = self.calculate_position_size(symbol, close, sl_price)
                
                if qty > 0:
                    cost = qty * close * (1 + self.COMMISSION)
                    if cost <= self.equity[symbol]:
                        if self.execute_buy(symbol, close, qty, sl_price):
                            self.equity[symbol] -= cost
                            self.positions[symbol] = {
                                'qty': qty,
                                'entry_price': close,
                                'sl_price': sl_price,
                                'entry_time': datetime.now()
                            }
                            self.log_trade(symbol, 'buy', close, qty, '', 0)
    
    def log_trade(self, symbol, side, price, qty, reason, pnl):
        """Registra un trade en CSV"""
        log_file = 'trades_ema.csv'
        
        trade_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'price': price,
            'qty': qty,
            'reason': reason,
            'pnl': pnl,
            'equity': self.equity[symbol]
        }
        
        df = pd.DataFrame([trade_data])
        
        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)
    
    def save_state(self):
        """Guarda el estado del bot"""
        state = {
            'equity': self.equity,
            'positions': {k: v for k, v in self.positions.items() if v is not None},
            'last_update': datetime.now().isoformat()
        }
        
        with open('bot_state_ema.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self):
        """Carga el estado del bot"""
        if os.path.exists('bot_state_ema.json'):
            with open('bot_state_ema.json', 'r') as f:
                state = json.load(f)
                self.equity = state.get('equity', self.equity)
                
                # Cargar posiciones
                for symbol, pos in state.get('positions', {}).items():
                    if pos and 'entry_time' in pos:
                        pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
                    self.positions[symbol] = pos
                
                print(f"✅ Estado cargado desde bot_state_ema.json")
    
    def run_once(self):
        """Ejecuta un ciclo de análisis"""
        print(f"\n{'='*60}")
        print(f"🔄 Ciclo de análisis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        for symbol in self.symbols:
            print(f"\n📊 Procesando {symbol}...")
            self.process_symbol(symbol)
        
        # Guardar estado
        self.save_state()
        
        # Mostrar resumen
        total_equity = sum(self.equity.values())
        initial_capital = self.capital_per_pair * len(self.symbols)
        roi = ((total_equity - initial_capital) / initial_capital) * 100
        
        print(f"\n{'='*60}")
        print(f"💰 RESUMEN")
        print(f"{'='*60}")
        print(f"Equity Total: ${total_equity:.2f}")
        print(f"ROI: {roi:+.2f}%")
        print(f"Posiciones abiertas: {sum(1 for p in self.positions.values() if p is not None)}/{len(self.symbols)}")
        print(f"{'='*60}\n")
    
    def run_continuous(self, interval_hours=4):
        """Ejecuta el bot continuamente sincronizado con velas de 4h"""
        print(f"🚀 Iniciando bot en modo continuo (cada {interval_hours}h)")
        print(f"⏰ Sincronizando con cierre de velas...\n")
        
        while True:
            try:
                # Calcular próximo cierre de vela
                now = datetime.utcnow()
                current_hour = now.hour
                next_close_hour = ((current_hour // interval_hours) + 1) * interval_hours
                
                if next_close_hour >= 24:
                    next_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_close = now.replace(hour=next_close_hour, minute=0, second=0, microsecond=0)
                
                # Añadir 30 segundos de delay para asegurar datos completos
                next_run = next_close + timedelta(seconds=30)
                
                wait_seconds = (next_run - now).total_seconds()
                
                if wait_seconds > 0:
                    print(f"⏳ Esperando {wait_seconds/60:.1f} minutos hasta próximo análisis...")
                    print(f"📅 Próxima ejecución: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
                    time.sleep(wait_seconds)
                
                # Ejecutar análisis
                self.run_once()
                
            except KeyboardInterrupt:
                print("\n⚠️ Bot detenido por el usuario")
                break
            except Exception as e:
                print(f"\n❌ Error en el ciclo: {e}")
                self.telegram.notify_error(f"Error en bot EMA: {str(e)}")
                time.sleep(300)  # Esperar 5 minutos antes de reintentar


if __name__ == '__main__':
    # Leer modo desde .env
    MODE = os.getenv('TRADING_MODE', 'paper').lower()
    
    # Crear y ejecutar bot
    bot = EMABot(mode=MODE)
    bot.run_continuous()
