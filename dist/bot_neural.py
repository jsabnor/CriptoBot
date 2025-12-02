import os
from dotenv import load_dotenv
# Cargar variables de entorno ANTES de importar config
load_dotenv()



import time
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import json
from telegram_notifier import TelegramNotifier
from data_cache import DataCache
from neural_bot import NeuralStrategy
import config

# ============================================================================
# BOT NEURAL v1.0 - ESTRATEGIA DE REDES NEURONALES
# ============================================================================
# Usa modelos pre-entrenados (CNN-LSTM) para predecir movimientos
# Timeframe: 4h
# ============================================================================

class NeuralBot:
    def __init__(self, mode=None, model_name=None, bot_id=None):
        """
        Inicializa el bot de trading neuronal.
        
        Args:
            mode: Modo de operación (paper/live)
            model_name: Nombre del modelo a cargar (None = usar default)
            bot_id: Identificador único del bot (ej: BTC, ETH) para archivos de estado
        """
        # Cargar credenciales
        self.API_KEY = config.API_KEY
        self.API_SECRET = config.API_SECRET
        
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError("ERROR: Claves API no configuradas en .env")
        
        # Configuración
        self.MODE = mode if mode else config.TRADING_MODE
        self.TIMEFRAME = config.TIMEFRAME
        self.BOT_ID = bot_id if bot_id else "neural"
        
        # Override: Bot Neural usa solo top 3 performers (basado en backtest)
        # Si se especifica un modelo, intentamos deducir el símbolo o usamos el config
        # Pero para multi-instancia, el usuario debería pasar --symbols
        # Aquí mantenemos la lista por defecto pero se sobreescribirá si se pasa --symbols en main
        self.SYMBOLS = ['SOL/USDT', 'ETH/USDT', 'XRP/USDT'] 
        self.CAPITAL_PER_PAIR = config.CAPITAL_PER_PAIR
        self.TOTAL_CAPITAL = self.CAPITAL_PER_PAIR * len(self.SYMBOLS)
        
        # Estado
        self.positions = {}
        self.equity = {}
        self.trades_log = []
        self.last_summary_date = None
        
        # Archivos de estado DINÁMICOS
        self.STATE_FILE = f'bot_state_neural_{self.BOT_ID}.json'
        self.TRADES_FILE = f'trades_neural_{self.BOT_ID}.csv'
        
        # Inicializar equity
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
        
        # Componentes
        self.telegram = TelegramNotifier()
        self.data_cache = DataCache()
        
        # Estrategia Neuronal
        print(f"🧠 [{self.BOT_ID}] Cargando estrategia neuronal...")
        if model_name:
            print(f"   Modelo especificado: {model_name}")
        self.strategy = NeuralStrategy(model_name=model_name)
        if self.strategy.model is None:
            print("⚠️ ADVERTENCIA: No se pudo cargar el modelo neuronal.")
            print("   Asegúrate de tener modelos en la carpeta 'models/'")
        
        # Cargar estado previo
        self.load_state()
        
        print(f"\n{'='*70}")
        print(f"BOT NEURAL [{self.BOT_ID}] - MODO: {self.MODE.upper()}")
        print(f"{'='*70}")
        print(f"Timeframe: {self.TIMEFRAME}")
        print(f"Pares: {len(self.SYMBOLS)}")
        print(f"Modelo cargado: v{self.strategy.version if self.strategy.model else 'Ninguno'}")
        print(f"Telegram: {'✓ Habilitado' if self.telegram.enabled else '✗ Deshabilitado'}")
        print(f"Estado: {self.STATE_FILE}")
        print(f"{'='*70}\n")
        
        if self.telegram.enabled:
            self.telegram.notify_startup(self.MODE, self.SYMBOLS, self.TOTAL_CAPITAL, strategy_name=f"NEURAL-{self.BOT_ID}")

    def load_state(self):
        """Carga el estado del bot desde JSON."""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.equity = state.get('equity', self.equity)
                    self.positions = state.get('positions', self.positions)
                    self.last_summary_date = state.get('last_summary_date')
                print(f"📂 Estado cargado de {self.STATE_FILE}")
            except Exception as e:
                print(f"❌ Error cargando estado: {e}")

    def save_state(self):
        """Guarda el estado actual en JSON."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'equity': self.equity,
            'positions': self.positions,
            'last_summary_date': self.last_summary_date,
            'bot_id': self.BOT_ID
        }
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"❌ Error guardando estado: {e}")

    def log_trade(self, trade_data):
        """Registra un trade en CSV."""
        file_exists = os.path.isfile(self.TRADES_FILE)
        try:
            df = pd.DataFrame([trade_data])
            df.to_csv(self.TRADES_FILE, mode='a', header=not file_exists, index=False)
        except Exception as e:
            print(f"❌ Error guardando trade: {e}")

    def get_current_price(self, symbol):
        """Obtiene precio actual (ticker)."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ Error precio {symbol}: {e}")
            return None

    def execute_buy(self, symbol, price, signal_data):
        """Ejecuta orden de compra basada en señal neuronal."""
        capital = self.equity.get(symbol, 0) # Use get to avoid key error if symbol not in equity
        if capital < config.MIN_EQUITY:
            print(f"⚠️ Capital insuficiente en {symbol}: ${capital:.2f}")
            return False
            
        # Calcular cantidad (usando 98% del capital para dejar margen para comisiones)
        qty = (capital * 0.98) / price
        
        # Ajustar precisión
        try:
            market = self.exchange.market(symbol)
            qty = self.exchange.amount_to_precision(symbol, qty)
        except:
            pass # Si falla, usar qty tal cual (paper mode)
            
        print(f"🚀 BUY SIGNAL {symbol} @ ${price:.4f} | Confianza: {signal_data['confidence']:.2f}")
        
        if self.MODE == 'live':
            try:
                order = self.exchange.create_market_buy_order(symbol, qty)
                actual_price = order['average'] if 'average' in order else price
                cost = float(order['cost']) if 'cost' in order else (float(qty) * actual_price)
                print(f"✅ Orden ejecutada: {qty} {symbol} @ ${actual_price}")
            except Exception as e:
                print(f"❌ Error orden compra: {e}")
                return False
        else:
            actual_price = price
            cost = float(qty) * actual_price
            
        # Stop Loss y Take Profit (basados en ATR o fijos, aquí usaremos fijos por simplicidad inicial)
        # La estrategia neuronal pura a veces no usa SL/TP fijos, sino salida por señal contraria
        # Pero por seguridad añadiremos un SL básico del 5%
        sl_price = actual_price * 0.95
        
        self.positions[symbol] = {
            'type': 'long',
            'entry_price': actual_price,
            'qty': float(qty),
            'cost': cost,
            'sl_price': sl_price,
            'entry_time': datetime.now().isoformat(),
            'confidence': signal_data['confidence']
        }
        
        self.equity[symbol] -= cost # Restar efectivo
        self.save_state()
        
        # Notificar
        if self.telegram.enabled:
            self.telegram.notify_buy(
                symbol, actual_price, qty, cost, sl_price, 0, 
                strategy_name=f"NEURAL-{self.BOT_ID} (Conf: {signal_data['confidence']:.2f})"
            )
            
        return True

    def execute_sell(self, symbol, price, reason="SIGNAL"):
        """Ejecuta orden de venta."""
        pos = self.positions.get(symbol)
        if not pos:
            return False
            
        qty = pos['qty']
        
        print(f"📉 SELL SIGNAL {symbol} @ ${price:.4f} ({reason})")
        
        if self.MODE == 'live':
            try:
                order = self.exchange.create_market_sell_order(symbol, qty)
                actual_price = order['average'] if 'average' in order else price
                revenue = float(order['cost']) if 'cost' in order else (float(qty) * actual_price)
            except Exception as e:
                print(f"❌ Error orden venta: {e}")
                return False
        else:
            actual_price = price
            revenue = float(qty) * actual_price
            
        # Calcular PnL
        cost = pos['cost']
        pnl = revenue - cost
        pnl_percent = (pnl / cost) * 100
        
        # Actualizar equity
        self.equity[symbol] += revenue
        self.positions[symbol] = None
        self.save_state()
        
        # Log
        self.log_trade({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'type': 'SELL',
            'reason': reason,
            'entry_price': pos['entry_price'],
            'exit_price': actual_price,
            'qty': qty,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'duration': 'N/A' # TODO: Calcular duración
        })
        
        print(f"✅ Venta completada. PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        # Notificar
        if self.telegram.enabled:
            self.telegram.notify_sell(
                symbol, actual_price, qty, revenue, pnl, pnl_percent, 
                reason, strategy_name=f"NEURAL-{self.BOT_ID}"
            )
            
        return True

    def run_analysis(self):
        """Ejecuta análisis de mercado."""
        print(f"\n🔍 [{self.BOT_ID}] Analizando mercado {datetime.now().strftime('%H:%M:%S')}...")
        
        for symbol in self.SYMBOLS:
            try:
                # Obtener señal neuronal
                signal_data = self.strategy.get_signal(symbol, self.TIMEFRAME)
                
                signal = signal_data.get('signal')
                confidence = signal_data.get('confidence', 0)
                current_price = self.get_current_price(symbol)
                
                if not current_price:
                    continue
                    
                print(f"  {symbol}: {signal} (Conf: {confidence:.2f}) - Precio: ${current_price:.2f}")
                
                pos = self.positions.get(symbol)
                
                # Lógica de Trading
                if pos:
                    # Gestión de posición abierta
                    # 1. Stop Loss de seguridad
                    if current_price <= pos['sl_price']:
                        self.execute_sell(symbol, current_price, "STOP_LOSS")
                    
                    # 2. Señal de Venta Neuronal
                    elif signal == 'SELL':
                        self.execute_sell(symbol, current_price, "NEURAL_SELL")
                        
                else:
                    # Buscar entrada
                    if signal == 'BUY':
                        self.execute_buy(symbol, current_price, signal_data)
                        
            except Exception as e:
                print(f"❌ Error analizando {symbol}: {e}")

    def send_daily_summary(self):
        """Envía resumen diario."""
        # Lógica simplificada para resumen diario
        # (Se puede copiar la lógica completa de bot_production.py si se desea detalle)
        pass

    def run_continuous(self):
        """Bucle principal de ejecución."""
        print(f"🚀 [{self.BOT_ID}] Iniciando bucle continuo...", flush=True)
        
        while True:
            try:
                now = datetime.utcnow()
                
                # Sincronizar con velas de 4h (00, 04, 08, 12, 16, 20 UTC)
                next_hour = now.hour + (4 - now.hour % 4)
                if next_hour >= 24:
                    next_hour = 0
                    # next_day logic omitted for brevity, just wait
                
                # Calcular segundos para esperar
                # Por simplicidad en este script, ejecutamos cada minuto para pruebas
                # En producción real, descomentar la espera larga
                
                self.run_analysis()
                
                # Esperar 1 minuto antes de siguiente chequeo (para no saturar)
                # En producción real, esto debería esperar al cierre de vela
                print("⏳ Esperando 60s...", flush=True)
                time.sleep(60)
                
            except KeyboardInterrupt:
                print(f"\n🛑 Deteniendo bot {self.BOT_ID}...")
                break
            except Exception as e:
                print(f"❌ Error en bucle principal: {e}")
                time.sleep(60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Trading Bot')
    parser.add_argument('--mode', type=str, choices=['paper', 'live'], help='Trading mode')
    parser.add_argument('--model', type=str, help='Model name to load')
    parser.add_argument('--id', type=str, help='Unique Bot ID (e.g. BTC, ETH)', required=True)
    parser.add_argument('--symbols', type=str, help='Comma separated symbols (e.g. BTC/USDT)')
    args = parser.parse_args()
    
    try:
        # Obtener model_name de argumentos o variable de entorno
        model_name = args.model or os.getenv('NEURAL_MODEL')
        
        bot = NeuralBot(mode=args.mode, model_name=model_name, bot_id=args.id)
        
        # Override symbols if provided
        if args.symbols:
            bot.SYMBOLS = [s.strip() for s in args.symbols.split(',')]
            # Recalculate capital per pair if needed, or just warn
            print(f"ℹ️ Sobreescribiendo símbolos: {bot.SYMBOLS}")
            
        bot.run_continuous()
    except Exception as e:
        print(f"❌ Error fatal: {e}")
