#!/usr/bin/env python3
"""
Bot de Telegram Interactivo para Consultas de Trading

Permite consultar el estado de los bots ADX, EMA y Neural mediante comandos.
Incluye reportes detallados de rendimiento y gestión de posiciones.
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
                    {'text': '💼 Posiciones Abiertas', 'callback_data': 'positions'},
                    {'text': '📊 Reportes y Ganancias', 'callback_data': 'reports_menu'}
                ],
                [
                    {'text': '⚙️ Estado del Sistema', 'callback_data': 'status'},
                    {'text': '❓ Ayuda', 'callback_data': 'help'}
                ]
            ]
        }

    def get_reports_keyboard(self):
        """Genera el teclado de selección de reportes"""
        return {
            'inline_keyboard': [
                [
                    {'text': '🤖 Reporte ADX', 'callback_data': 'report_adx'},
                    {'text': '📉 Reporte EMA', 'callback_data': 'report_ema'}
                ],
                [
                    {'text': '🧠 Reporte Neural', 'callback_data': 'report_neural'},
                    {'text': '🔙 Volver al Menú', 'callback_data': 'main_menu'}
                ]
            ]
        }
    
    def get_back_keyboard(self, menu='main'):
        """Genera botón de volver"""
        callback = 'main_menu' if menu == 'main' else 'reports_menu'
        return {
            'inline_keyboard': [[{'text': '🔙 Volver', 'callback_data': callback}]]
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
    
    def get_trades_df(self, bot_name='adx'):
        """Lee el DataFrame de trades de un bot"""
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
            # Intentar parsear timestamp con manejo de errores
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass
            return df
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
        """Calcula el equity total de un bot"""
        if not bot_state:
            return 0, 0, 0
        
        equity_dict = bot_state.get('equity', {})
        equity_cash = sum(equity_dict.values())
        
        positions_value = 0
        positions = bot_state.get('positions', {})
        
        for symbol, position in positions.items():
            if position:
                current_price = self.get_current_price(symbol)
                if current_price:
                    qty = position.get('size', position.get('qty', 0))
                    if qty > 0:
                        positions_value += qty * current_price
        
        equity_total = equity_cash + positions_value
        return equity_total, equity_cash, positions_value
    
    def cmd_start(self, chat_id):
        """Comando /start - Menú principal"""
        text = (
            "🤖 <b>Bot de Trading - Panel de Control</b>\n\n"
            "Selecciona una opción:\n\n"
            "• <b>Posiciones Abiertas</b>: Ver operaciones en curso y P&L flotante.\n"
            "• <b>Reportes y Ganancias</b>: Historial detallado y beneficios por bot/par.\n"
            "• <b>Estado del Sistema</b>: Salud y capital total de los bots."
        )
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_help(self, chat_id):
        """Comando /help - Ayuda"""
        text = (
            "📚 <b>Ayuda</b>\n\n"
            "Usa los botones del menú para navegar.\n\n"
            "<b>Comandos disponibles:</b>\n"
            "/start - Menú principal\n"
            "/posiciones - Ver operaciones abiertas\n"
            "/status - Estado general\n"
            "/reporte [adx|ema|neural] - Reporte rápido"
        )
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_status(self, chat_id):
        """Comando /status - Estado de los bots"""
        text = "⚙️ <b>ESTADO DEL SISTEMA</b>\n\n"
        
        total_combined = 0
        
        for bot_name, label in [('adx', '🤖 ADX'), ('ema', '📉 EMA'), ('neural', '🧠 Neural')]:
            state = self.get_bot_state(bot_name)
            if state:
                equity, _, _ = self.calculate_total_equity(state)
                total_combined += equity
                
                last_update = state.get('timestamp', state.get('last_update', state.get('last_summary_date', 'N/A')))
                if isinstance(last_update, str):
                    last_update = last_update.split('T')[0]
                
                text += f"<b>{label}</b>: ✅ Activo\n"
                text += f"  💰 Equity: ${equity:.2f}\n"
                text += f"  📅 Update: {last_update}\n\n"
            else:
                text += f"<b>{label}</b>: ❌ Inactivo/Sin datos\n\n"
        
        text += f"━━━━━━━━━━━━━━━━\n"
        text += f"💼 <b>CAPITAL TOTAL: ${total_combined:.2f}</b>"
        
        self.send_message(chat_id, text, self.get_main_keyboard())
    
    def cmd_positions(self, chat_id):
        """Comando /posiciones - Muestra posiciones abiertas con cálculos"""
        text = "💼 <b>POSICIONES ABIERTAS</b>\n"
        has_positions = False
        
        for bot_name, label in [('adx', '🤖 ADX'), ('ema', '📉 EMA'), ('neural', '🧠 Neural')]:
            state = self.get_bot_state(bot_name)
            if not state:
                continue
                
            positions = state.get('positions', {})
            bot_has_pos = False
            
            bot_text = f"\n<b>{label}:</b>\n"
            
            for symbol, pos in positions.items():
                if pos:
                    has_positions = True
                    bot_has_pos = True
                    
                    # Normalizar datos (diferentes formatos según bot)
                    entry_price = pos.get('entry_price', 0)
                    qty = pos.get('size', pos.get('qty', 0))
                    
                    current_price = self.get_current_price(symbol)
                    
                    sym_clean = symbol.replace('/USDT', '')
                    text_pos = f"  🪙 <b>{sym_clean}</b>"
                    
                    if current_price:
                        pnl = (current_price - entry_price) * qty
                        roi = ((current_price - entry_price) / entry_price) * 100
                        emoji = '🟢' if pnl >= 0 else '🔴'
                        text_pos += f" {emoji} ${pnl:+.2f} ({roi:+.2f}%)"
                        text_pos += f"\n     Entrada: ${entry_price:.4f} | Actual: ${current_price:.4f}"
                    else:
                        text_pos += f"\n     Entrada: ${entry_price:.4f} | Qty: {qty:.4f}"
                    
                    bot_text += text_pos + "\n"
            
            if bot_has_pos:
                text += bot_text
        
        if not has_positions:
            text += "\n📭 No hay operaciones abiertas actualmente."
        
        self.send_message(chat_id, text, self.get_main_keyboard())

    def cmd_reports_menu(self, chat_id):
        """Muestra menú de selección de reportes"""
        text = "📊 <b>REPORTES Y GANANCIAS</b>\n\nSelecciona el bot para ver su historial detallado y beneficios:"
        self.send_message(chat_id, text, self.get_reports_keyboard())

    def generate_bot_report(self, bot_name):
        """Genera reporte detallado para un bot"""
        df = self.get_trades_df(bot_name)
        
        label_map = {'adx': '🤖 ADX Bot', 'ema': '📉 EMA Bot', 'neural': '🧠 Neural Bot'}
        label = label_map.get(bot_name, bot_name.upper())
        
        if df is None or df.empty:
            return f"<b>{label}</b>\n\n❌ No hay historial de operaciones registrado."
        
        # Normalizar columnas
        # ADX/EMA usan 'pnl', Neural usa 'pnl' (pero a veces 'profit' en backtest, aquí es prod)
        # Neural usa 'type'='SELL' para cierres. ADX 'type'='sell'. EMA 'side'='sell'.
        
        # Filtrar solo ventas/cierres (donde se realiza el PnL)
        if 'side' in df.columns: # EMA
            closed_trades = df[df['side'] == 'sell']
        elif 'type' in df.columns: # ADX y Neural
            closed_trades = df[df['type'].str.upper() == 'SELL']
        else:
            closed_trades = pd.DataFrame()
            
        if closed_trades.empty:
            return f"<b>{label}</b>\n\nℹ️ Se han registrado entradas pero ninguna operación cerrada aún."
            
        # Estadísticas Generales
        total_pnl = closed_trades['pnl'].sum()
        total_trades = len(closed_trades)
        wins = len(closed_trades[closed_trades['pnl'] > 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        text = f"<b>{label} - REPORTE DE RENDIMIENTO</b>\n"
        text += f"━━━━━━━━━━━━━━━━\n"
        text += f"💰 <b>Beneficio Total: ${total_pnl:+.2f}</b>\n"
        text += f"🔢 Operaciones: {total_trades}\n"
        text += f"🎯 Win Rate: {win_rate:.1f}%\n\n"
        
        text += f"<b>📊 DESGLOSE POR PAR:</b>\n"
        
        # Agrupar por símbolo
        grouped = closed_trades.groupby('symbol')
        
        for symbol, group in grouped:
            sym_pnl = group['pnl'].sum()
            sym_trades = len(group)
            sym_wins = len(group[group['pnl'] > 0])
            sym_wr = (sym_wins / sym_trades * 100) if sym_trades > 0 else 0
            
            sym_clean = symbol.replace('/USDT', '')
            emoji = '🟢' if sym_pnl >= 0 else '🔴'
            
            text += f"\n🪙 <b>{sym_clean}</b>: {emoji} <b>${sym_pnl:+.2f}</b>\n"
            text += f"   └ Trades: {sym_trades} (WR: {sym_wr:.0f}%)\n"
            
            # Última operación del par
            last_trade = group.iloc[-1]
            last_date = last_trade['timestamp'].strftime('%d/%m')
            text += f"   └ Última: {last_date} (${last_trade['pnl']:+.2f})\n"

        return text

    def cmd_bot_report(self, chat_id, bot_name):
        """Envía el reporte de un bot específico"""
        text = self.generate_bot_report(bot_name)
        self.send_message(chat_id, text, self.get_back_keyboard('reports'))

    def handle_callback_query(self, callback_query):
        """Maneja callbacks de botones inline"""
        chat_id = callback_query['message']['chat']['id']
        data = callback_query['data']
        
        # Mapeo de callbacks
        if data == 'main_menu':
            self.cmd_start(chat_id)
        elif data == 'status':
            self.cmd_status(chat_id)
        elif data == 'positions':
            self.cmd_positions(chat_id)
        elif data == 'help':
            self.cmd_help(chat_id)
        elif data == 'reports_menu':
            self.cmd_reports_menu(chat_id)
        elif data.startswith('report_'):
            bot_name = data.split('_')[1]
            self.cmd_bot_report(chat_id, bot_name)
        
        # Responder al callback
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
        
        if not self.is_authorized(chat_id):
            self.send_message(chat_id, "🚫 Acceso Denegado")
            return
        
        if text.startswith('/'):
            cmd = text.split()[0].lower()
            if cmd == '/start': self.cmd_start(chat_id)
            elif cmd == '/help': self.cmd_help(chat_id)
            elif cmd == '/status': self.cmd_status(chat_id)
            elif cmd == '/posiciones': self.cmd_positions(chat_id)
            elif cmd.startswith('/reporte'):
                parts = text.split()
                if len(parts) > 1:
                    self.cmd_bot_report(chat_id, parts[1].lower())
                else:
                    self.cmd_reports_menu(chat_id)
            else:
                self.send_message(chat_id, "❌ Comando desconocido")

    def get_updates(self):
        """Obtiene actualizaciones pendientes"""
        try:
            response = requests.get(
                f"{self.api_url}/getUpdates",
                params={'offset': self.last_update_id + 1, 'timeout': 30},
                timeout=35
            )
            if response.status_code == 200:
                return response.json().get('result', [])
        except Exception as e:
            print(f"❌ Error updates: {e}")
        return []
    
    def run(self):
        """Ejecuta el bot"""
        print("🚀 Bot Telegram Interactivo Iniciado")
        while True:
            try:
                updates = self.get_updates()
                for update in updates:
                    self.last_update_id = update['update_id']
                    
                    if 'message' in update:
                        self.handle_message(update['message'])
                    elif 'channel_post' in update:
                        self.handle_message(update['channel_post'])
                    elif 'callback_query' in update:
                        self.handle_callback_query(update['callback_query'])
                        
                time.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error loop: {e}")
                time.sleep(5)

if __name__ == '__main__':
    bot = TelegramBotHandler()
    bot.run()
