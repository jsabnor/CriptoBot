#!/usr/bin/env python3
"""
Bot de Telegram Interactivo para Consultas de Trading

Permite consultar el estado de los bots ADX y EMA mediante comandos.

Comandos disponibles:
- /start - Menú principal
- /help - Lista de comandos
- /status - Estado de ambos bots
- /posiciones - Posiciones abiertas
- /resumen - Resumen diario
- /historial [bot] [dias] - Historial de operaciones

Seguridad: Solo usuarios autorizados (TELEGRAM_AUTHORIZED_USERS)
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import time

load_dotenv()


class TelegramBotHandler:
    """Manejador de comandos interactivos de Telegram"""
    
    def __init__(self):
        """Inicializa el bot de comandos"""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.authorized_users = self._load_authorized_users()
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN no configurado en .env")
        
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        
        print("🤖 Bot de Telegram Interactivo iniciado")
        print(f"✅ Usuarios autorizados: {len(self.authorized_users)}")
    
    def _load_authorized_users(self):
        """Carga lista de usuarios autorizados"""
        users_str = os.getenv('TELEGRAM_AUTHORIZED_USERS', '')
        
        if not users_str:
            # Si no hay lista, usar el CHAT_ID principal
            main_chat = os.getenv('TELEGRAM_CHAT_ID', '')
            return [main_chat] if main_chat else []
        
        # Soporta formato: "12345,67890,111213"
        return [user.strip() for user in users_str.split(',') if user.strip()]
    
    def is_authorized(self, chat_id):
        """Verifica si el usuario está autorizado"""
        return str(chat_id) in self.authorized_users
    
    def send_message(self, chat_id, text, reply_markup=None):
        """Envía un mensaje a un chat específico"""
        try:
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'HTML'
            }
            
            if reply_markup:
                payload['reply_markup'] = reply_markup
            
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error enviando mensaje: {e}")
            return False
    
    def get_main_keyboard(self):
        """Genera el teclado principal con botones"""
        return {
            'inline_keyboard': [
                [
                    {'text': '📊 Estado', 'callback_data': 'status'},
                    {'text': '💼 Posiciones', 'callback_data': 'positions'}
                ],
                [
                    {'text': '📈 Resumen Diario', 'callback_data': 'summary'},
                    {'text': '📋 Historial', 'callback_data': 'history'}
                ],
                [
                    {'text': '🤖 Bot ADX', 'callback_data': 'adx_info'},
                    {'text': '📉 Bot EMA', 'callback_data': 'ema_info'}
                ],
                [
                    {'text': '❓ Ayuda', 'callback_data': 'help'}
                ]
            ]
        }
    def get_bot_state(self, bot_name='adx'):
        """Lee el estado de un bot desde su archivo JSON"""
        try:
            file_map = {
                'adx': 'bot_state.json',
                'ema': 'bot_state_ema.json',
                'neural': 'bot_state_neural.json'
            }
            
            filename = file_map.get(bot_name, 'bot_state.json')
            
            if not os.path.exists(filename):
                return None
            
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error leyendo estado de {bot_name}: {e}")
            return None
    
    def get_trades_history(self, bot_name='adx', days=7):
        """Lee el historial de trades de un bot"""
        try:
            file_map = {
                'adx': 'trades_production.csv',
                'ema': 'trades_ema.csv',
                'neural': 'trades_neural.csv'
            }
            
            filename = file_map.get(bot_name, 'trades_production.csv')
            
            if not os.path.exists(filename):
                return None
            
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filtrar últimos N días
            cutoff_date = datetime.now() - timedelta(days=days)
            df_recent = df[df['timestamp'] >= cutoff_date]
            
            return df_recent
        except Exception as e:
            print(f"❌ Error leyendo historial de {bot_name}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error leyendo historial de {bot_name}: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Obtiene el precio actual de un símbolo desde Binance"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.replace('/', '')}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            print(f"❌ Error obteniendo precio de {symbol}: {e}")
        return None
    
    def calculate_total_equity(self, bot_state):
        """
        Calcula el equity total de un bot (efectivo + valor de posiciones abiertas)
        
        Args:
            bot_state: Estado del bot desde JSON
            
        Returns:
            tuple: (equity_total, equity_cash, positions_value)
        """
        if not bot_state:
            return 0, 0, 0
        
        # Equity en efectivo
        equity_dict = bot_state.get('equity', {})
        equity_cash = sum(equity_dict.values())
        
        # Valor de posiciones abiertas
        positions_value = 0
        positions = bot_state.get('positions', {})
        
        for symbol, position in positions.items():
            if position:
                current_price = self.get_current_price(symbol)
                if current_price:
                    # Obtener cantidad según el formato del bot
                    qty = position.get('size', position.get('qty', 0))
                    if qty > 0:
                        positions_value += qty * current_price
        
        equity_total = equity_cash + positions_value
        
        return equity_total, equity_cash, positions_value
    
    def cmd_start(self, chat_id):
        """Comando /start - Menú principal"""
        text = (
            "🤖 <b>Bot de Trading - Panel de Control</b>\n\n"
            "Bienvenido al sistema de consultas del bot de trading.\n\n"
            "Usa los botones de abajo para consultar información:\n\n"
            "• <b>Estado</b>: Ver estado de ambos bots\n"
            "• <b>Posiciones</b>: Posiciones abiertas actuales\n"
            "• <b>Resumen</b>: Resumen diario de operaciones\n"
            "• <b>Historial</b>: Últimas operaciones realizadas\n"
            "• <b>Bot ADX/EMA</b>: Info específica de cada bot\n\n"
            "También puedes usar comandos de texto:\n"
            "/status, /posiciones, /resumen, /historial"
        )
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_help(self, chat_id):
        """Comando /help - Ayuda"""
        text = (
            "📚 <b>Comandos Disponibles</b>\n\n"
            "<b>Comandos básicos:</b>\n"
            "• /start - Menú principal\n"
            "• /help - Esta ayuda\n"
            "• /status - Estado de ambos bots\n"
            "• /posiciones - Ver posiciones abiertas\n"
            "• /resumen - Resumen del día\n"
            "• /historial - Últimas 10 operaciones\n\n"
            "<b>Comandos avanzados:</b>\n"
            "• /historial adx 7 - Historial bot ADX (7 días)\n"
            "• /historial ema 3 - Historial bot EMA (3 días)\n\n"
            "<b>Bots específicos:</b>\n"
            "• /adx - Info del bot ADX\n"
            "• /ema - Info del bot EMA"
        )
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_status(self, chat_id):
        """Comando /status - Estado de ambos bots"""
        # Leer estados
        adx_state = self.get_bot_state('adx')
        ema_state = self.get_bot_state('ema')
        neural_state = self.get_bot_state('neural')
        
        if not adx_state and not ema_state and not neural_state:
            text = "❌ No se pudo leer el estado de los bots"
            self.send_message(chat_id, text)
            return
        
        text = "📊 <b>ESTADO DE LOS BOTS</b>\n\n"
        
        total_equity_adx = 0
        total_equity_ema = 0
        total_equity_neural = 0
        
        # Bot ADX
        if adx_state:
            equity_total, equity_cash, positions_value = self.calculate_total_equity(adx_state)
            total_equity_adx = equity_total
            positions_adx = sum(1 for p in adx_state.get('positions', {}).values() if p)
            
            text += (
                "🤖 <b>Bot ADX (Estrategia ADX + ATR)</b>\n"
                f"💰 Equity Total: <b>${equity_total:.2f}</b>\n"
            )
            
            # Mostrar desglose si hay posiciones
            if positions_value > 0:
                text += (
                    f"  └ Efectivo: ${equity_cash:.2f}\n"
                    f"  └ Posiciones: ${positions_value:.2f}\n"
                )
            
            text += (
                f"📍 Posiciones: <b>{positions_adx}/4</b>\n"
                f"📅 Última actualización: {adx_state.get('timestamp', 'N/A')}\n\n"
            )
        else:
            text += "🤖 <b>Bot ADX</b>: ❌ Estado no disponible\n\n"
        
        # Bot EMA
        if ema_state:
            equity_total, equity_cash, positions_value = self.calculate_total_equity(ema_state)
            total_equity_ema = equity_total
            positions_ema = sum(1 for p in ema_state.get('positions', {}).values() if p)
            
            text += (
                "📉 <b>Bot EMA (Estrategia EMA 15/30)</b>\n"
                f"💰 Equity Total: <b>${equity_total:.2f}</b>\n"
            )
            
            # Mostrar desglose si hay posiciones
            if positions_value > 0:
                text += (
                    f"  └ Efectivo: ${equity_cash:.2f}\n"
                    f"  └ Posiciones: ${positions_value:.2f}\n"
                )
            
            text += (
                f"📍 Posiciones: <b>{positions_ema}/4</b>\n"
                f"📅 Última actualización: {ema_state.get('last_update', 'N/A')}\n\n"
            )
        else:
            text += "📉 <b>Bot EMA</b>: ❌ Estado no disponible\n\n"
            
        # Bot Neural
        if neural_state:
            equity_total, equity_cash, positions_value = self.calculate_total_equity(neural_state)
            total_equity_neural = equity_total
            positions_neural = sum(1 for p in neural_state.get('positions', {}).values() if p)
            
            text += (
                "🧠 <b>Bot Neural (CNN-LSTM)</b>\n"
                f"💰 Equity Total: <b>${equity_total:.2f}</b>\n"
            )
            
            # Mostrar desglose si hay posiciones
            if positions_value > 0:
                text += (
                    f"  └ Efectivo: ${equity_cash:.2f}\n"
                    f"  └ Posiciones: ${positions_value:.2f}\n"
                )
            
            text += (
                f"📍 Posiciones: <b>{positions_neural}/4</b>\n"
                f"📅 Última actualización: {neural_state.get('last_summary_date', 'N/A')}\n\n"
            )
        else:
            text += "🧠 <b>Bot Neural</b>: ❌ Estado no disponible\n\n"
        
        # Total combinado
        if adx_state or ema_state or neural_state:
            total_combined = total_equity_adx + total_equity_ema + total_equity_neural
            text += (
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 <b>EQUITY TOTAL: ${total_combined:.2f}</b>"
            )
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_positions(self, chat_id):
        """Comando /posiciones - Posiciones abiertas"""
        adx_state = self.get_bot_state('adx')
        ema_state = self.get_bot_state('ema')
        neural_state = self.get_bot_state('neural')
        
        text = "💼 <b>POSICIONES ABIERTAS</b>\n\n"
        has_positions = False
        
        # Posiciones ADX
        if adx_state:
            text += "🤖 <b>Bot ADX:</b>\n"
            positions = adx_state.get('positions', {})
            
            for symbol, pos in positions.items():
                if pos:
                    has_positions = True
                    entry_price = pos.get('entry_price', 0)
                    size = pos.get('size', 0)
                    sl_price = pos.get('sl_price', 0)
                    entry_time_str = pos.get('entry_time')
                    
                    # Obtener precio actual
                    current_price = self.get_current_price(symbol)
                    
                    text += f"\n🪙 <b>{symbol.replace('/USDT', '')}</b>\n"
                    text += f"  ├ Entrada: ${entry_price:.4f}\n"
                    
                    # Mostrar precio actual y P&L si se pudo obtener
                    if current_price:
                        pnl = (current_price - entry_price) * size
                        roi = ((current_price - entry_price) / entry_price) * 100
                        pnl_emoji = '💚' if pnl >= 0 else '💔'
                        
                        text += f"  ├ Actual: <b>${current_price:.4f}</b>\n"
                        text += f"  ├ {pnl_emoji} P&L: <b>${pnl:+.2f}</b> ({roi:+.2f}%)\n"
                    
                    text += f"  ├ Cantidad: {size:.6f}\n"
                    text += f"  ├ Stop Loss: ${sl_price:.4f}\n"
                    
                    # Calcular duración
                    if entry_time_str:
                        try:
                            from datetime import datetime
                            if isinstance(entry_time_str, str):
                                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                            else:
                                entry_time = entry_time_str
                            
                            duration = datetime.now() - entry_time.replace(tzinfo=None)
                            hours = duration.total_seconds() / 3600
                            
                            if hours < 1:
                                duration_str = f"{int(duration.total_seconds() / 60)}m"
                            elif hours < 24:
                                duration_str = f"{int(hours)}h {int((hours % 1) * 60)}m"
                            else:
                                days = int(hours / 24)
                                remaining_hours = int(hours % 24)
                                duration_str = f"{days}d {remaining_hours}h"
                            
                            text += f"  └ ⏱️ Duración: {duration_str}\n"
                        except:
                            pass
            
            if not any(positions.values()):
                text += "  └ Sin posiciones abiertas\n"
            
            text += "\n"
        
        # Posiciones EMA
        if ema_state:
            text += "📉 <b>Bot EMA:</b>\n"
            positions = ema_state.get('positions', {})
            
            for symbol, pos in positions.items():
                if pos:
                    has_positions = True
                    entry_price = pos.get('entry_price', 0)
                    qty = pos.get('qty', 0)
                    sl_price = pos.get('sl_price', 0)
                    entry_time_str = pos.get('entry_time')
                    
                    # Obtener precio actual
                    current_price = self.get_current_price(symbol)
                    
                    text += f"\n🪙 <b>{symbol.replace('/USDT', '')}</b>\n"
                    text += f"  ├ Entrada: ${entry_price:.4f}\n"
                    
                    # Mostrar precio actual y P&L si se pudo obtener
                    if current_price:
                        pnl = (current_price - entry_price) * qty
                        roi = ((current_price - entry_price) / entry_price) * 100
                        pnl_emoji = '💚' if pnl >= 0 else '💔'
                        
                        text += f"  ├ Actual: <b>${current_price:.4f}</b>\n"
                        text += f"  ├ {pnl_emoji} P&L: <b>${pnl:+.2f}</b> ({roi:+.2f}%)\n"
                    
                    text += f"  ├ Cantidad: {qty:.6f}\n"
                    text += f"  ├ Stop Loss: ${sl_price:.4f}\n"
                    
                    # Calcular duración
                    if entry_time_str:
                        try:
                            from datetime import datetime
                            if isinstance(entry_time_str, str):
                                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                            else:
                                entry_time = entry_time_str
                            
                            duration = datetime.now() - entry_time.replace(tzinfo=None)
                            hours = duration.total_seconds() / 3600
                            
                            if hours < 1:
                                duration_str = f"{int(duration.total_seconds() / 60)}m"
                            elif hours < 24:
                                duration_str = f"{int(hours)}h {int((hours % 1) * 60)}m"
                            else:
                                days = int(hours / 24)
                                remaining_hours = int(hours % 24)
                                duration_str = f"{days}d {remaining_hours}h"
                            
                            text += f"  └ ⏱️ Duración: {duration_str}\n"
                        except:
                            pass
            
            if not positions:
                text += "  └ Sin posiciones abiertas\n"
                
        # Posiciones Neural
        if neural_state:
            text += "\n🧠 <b>Bot Neural:</b>\n"
            positions = neural_state.get('positions', {})
            
            for symbol, pos in positions.items():
                if pos:
                    has_positions = True
                    entry_price = pos.get('entry_price', 0)
                    qty = pos.get('qty', 0)
                    sl_price = pos.get('sl_price', 0)
                    entry_time_str = pos.get('entry_time')
                    confidence = pos.get('confidence', 0)
                    
                    # Obtener precio actual
                    current_price = self.get_current_price(symbol)
                    
                    text += f"\n🪙 <b>{symbol.replace('/USDT', '')}</b>\n"
                    text += f"  ├ Entrada: ${entry_price:.4f}\n"
                    
                    # Mostrar precio actual y P&L si se pudo obtener
                    if current_price:
                        pnl = (current_price - entry_price) * qty
                        roi = ((current_price - entry_price) / entry_price) * 100
                        pnl_emoji = '💚' if pnl >= 0 else '💔'
                        
                        text += f"  ├ Actual: <b>${current_price:.4f}</b>\n"
                        text += f"  ├ {pnl_emoji} P&L: <b>${pnl:+.2f}</b> ({roi:+.2f}%)\n"
                    
                    text += f"  ├ Cantidad: {qty:.6f}\n"
                    text += f"  ├ Stop Loss: ${sl_price:.4f}\n"
                    text += f"  ├ Confianza: {confidence:.2f}\n"
                    
                    # Calcular duración
                    if entry_time_str:
                        try:
                            from datetime import datetime
                            if isinstance(entry_time_str, str):
                                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                            else:
                                entry_time = entry_time_str
                            
                            duration = datetime.now() - entry_time.replace(tzinfo=None)
                            hours = duration.total_seconds() / 3600
                            
                            if hours < 1:
                                duration_str = f"{int(duration.total_seconds() / 60)}m"
                            elif hours < 24:
                                duration_str = f"{int(hours)}h {int((hours % 1) * 60)}m"
                            else:
                                days = int(hours / 24)
                                remaining_hours = int(hours % 24)
                                duration_str = f"{days}d {remaining_hours}h"
                            
                            text += f"  └ ⏱️ Duración: {duration_str}\n"
                        except:
                            pass
            
            if not any(positions.values()):
                text += "  └ Sin posiciones abiertas\n"
        
        if not has_positions:
            text += "\n📭 No hay posiciones abiertas en este momento"
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_summary(self, chat_id):
        """Comando /resumen - Resumen diario"""
        text = "📈 <b>RESUMEN DIARIO</b>\n\n"
        
        # Resumen ADX
        df_adx = self.get_trades_history('adx', days=1)
        if df_adx is not None and not df_adx.empty:
            sells = df_adx[df_adx['type'] == 'sell']
            if not sells.empty:
                pnl = sells['pnl'].sum()
                trades = len(sells)
                wins = len(sells[sells['pnl'] > 0])
                
                text += (
                    f"🤖 <b>Bot ADX</b>\n"
                    f"  └ Trades: {trades}\n"
                    f"  └ Ganadas: {wins}/{trades}\n"
                    f"  └ P&L: <b>${pnl:+.2f}</b>\n\n"
                )
            else:
                text += "🤖 <b>Bot ADX</b>: Sin operaciones cerradas hoy\n\n"
        else:
            text += "🤖 <b>Bot ADX</b>: Sin datos\n\n"
        
        # Resumen EMA
        df_ema = self.get_trades_history('ema', days=1)
        if df_ema is not None and not df_ema.empty:
            sells = df_ema[df_ema['side'] == 'sell']
            if not sells.empty:
                pnl = sells['pnl'].sum()
                trades = len(sells)
                wins = len(sells[sells['pnl'] > 0])
                
                text += (
                    f"📉 <b>Bot EMA</b>\n"
                    f"  └ Trades: {trades}\n"
                    f"  └ Ganadas: {wins}/{trades}\n"
                    f"  └ P&L: <b>${pnl:+.2f}</b>\n\n"
                )
            else:
                text += "📉 <b>Bot EMA</b>: Sin operaciones cerradas hoy\n\n"
        else:
            text += "📉 <b>Bot EMA</b>: Sin datos\n\n"
        
        text += f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_history(self, chat_id, args=None):
        """Comando /historial - Últimas operaciones"""
        # Parsear argumentos: /historial [bot] [dias]
        bot_name = 'adx'
        days = 7
        
        if args:
            parts = args.strip().split()
            if len(parts) >= 1 and parts[0].lower() in ['adx', 'ema']:
                bot_name = parts[0].lower()
            if len(parts) >= 2 and parts[1].isdigit():
                days = int(parts[1])
        
        df = self.get_trades_history(bot_name, days)
        
        bot_emoji = '🤖' if bot_name == 'adx' else '📉'
        text = f"📋 <b>HISTORIAL {bot_name.upper()}</b> (últimos {days} días)\n\n"
        
        if df is None or df.empty:
            text += "Sin operaciones en este período"
            self.send_message(chat_id, text, self.get_main_keyboard())
            return
        
        # Mostrar últimas 10 operaciones
        df_recent = df.tail(10).sort_values('timestamp', ascending=False)
        
        for _, row in df_recent.iterrows():
            timestamp = row['timestamp'].strftime('%d/%m %H:%M')
            symbol = row['symbol'].replace('/USDT', '')
            side = row.get('type', row.get('side', 'N/A'))
            price = row['price']
            pnl = row.get('pnl', 0)
            
            side_emoji = '🟢' if side == 'buy' else '🔴'
            pnl_text = f"${pnl:+.2f}" if pnl != 0 else ""
            
            text += (
                f"{side_emoji} {timestamp} - <b>{symbol}</b>\n"
                f"  └ {side.upper()} @ ${price:.4f} {pnl_text}\n\n"
            )
        
        if len(df) > 10:
            text += f"\n📊 Mostrando 10 de {len(df)} operaciones"
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def handle_callback_query(self, callback_query):
        """Maneja callbacks de botones inline"""
        chat_id = callback_query['message']['chat']['id']
        data = callback_query['data']
        
        # Mapeo de callbacks a comandos
        callback_map = {
            'status': self.cmd_status,
            'positions': self.cmd_positions,
            'summary': self.cmd_summary,
            'history': self.cmd_history,
            'help': self.cmd_help,
            'adx_info': lambda cid: self.cmd_history(cid, 'adx 7'),
            'ema_info': lambda cid: self.cmd_history(cid, 'ema 7'),
        }
        
        handler = callback_map.get(data)
        if handler:
            handler(chat_id)
        
        # Responder al callback para quitar el "loading"
        try:
            requests.post(
                f"{self.api_url}/answerCallbackQuery",
                json={'callback_query_id': callback_query['id']},
                timeout=5
            )
        except:
            pass
    
    def handle_message(self, message):
        """Maneja mensajes entrantes"""
        chat_id = message['chat']['id']
        text = message.get('text', '')
        user_id = message.get('from', {}).get('id', 'unknown')
        
        print(f"📩 DEBUG: Mensaje recibido de ChatID: {chat_id} | UserID: {user_id} | Texto: {text}")
        
        # Verificar autorización
        if not self.is_authorized(chat_id):
            self.send_message(
                chat_id,
                "🚫 <b>Acceso Denegado</b>\n\n"
                "No tienes autorización para usar este bot.\n"
                f"Tu Chat ID: <code>{chat_id}</code>"
            )
            return
        
        text = message.get('text', '')
        
        # Comandos
        if text.startswith('/'):
            parts = text.split(maxsplit=1)
            # Manejar /comando@nombre_bot
            command = parts[0].lower().split('@')[0]
            args = parts[1] if len(parts) > 1 else None
            
            command_map = {
                '/start': lambda: self.cmd_start(chat_id),
                '/help': lambda: self.cmd_help(chat_id),
                '/status': lambda: self.cmd_status(chat_id),
                '/posiciones': lambda: self.cmd_positions(chat_id),
                '/resumen': lambda: self.cmd_summary(chat_id),
                '/historial': lambda: self.cmd_history(chat_id, args),
                '/adx': lambda: self.cmd_history(chat_id, 'adx 7'),
                '/ema': lambda: self.cmd_history(chat_id, 'ema 7'),
            }
            
            handler = command_map.get(command)
            if handler:
                handler()
            else:
                self.send_message(
                    chat_id,
                    f"❌ Comando desconocido: {command}\n\nUsa /help para ver comandos disponibles"
                )
    
    def get_updates(self):
        """Obtiene actualizaciones pendientes"""
        try:
            response = requests.get(
                f"{self.api_url}/getUpdates",
                params={
                    'offset': self.last_update_id + 1,
                    'timeout': 30
                },
                timeout=35
            )
            
            if response.status_code == 200:
                return response.json().get('result', [])
        except Exception as e:
            print(f"❌ Error obteniendo updates: {e}")
        
        return []
    
    def run(self):
        """Ejecuta el bot en modo polling"""
        print("🚀 Bot iniciado en modo polling...")
        print("🛑 Presiona Ctrl+C para detener\n")
        
        while True:
            try:
                updates = self.get_updates()
                
                for update in updates:
                    self.last_update_id = update['update_id']
                    
                    # Manejar mensajes (privados y grupos)
                    if 'message' in update:
                        self.handle_message(update['message'])
                    
                    # Manejar mensajes de canales
                    elif 'channel_post' in update:
                        self.handle_message(update['channel_post'])
                    
                    # Manejar callbacks de botones
                    elif 'callback_query' in update:
                        self.handle_callback_query(update['callback_query'])
                
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Bot detenido por el usuario")
                break
            except Exception as e:
                print(f"❌ Error en el ciclo principal: {e}")
                time.sleep(5)


if __name__ == '__main__':
    bot = TelegramBotHandler()
    bot.run()
