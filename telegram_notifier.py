import os
import requests
from datetime import datetime

class TelegramNotifier:
    """
    Gestor de notificaciones de Telegram para el bot de trading.
    Usa la API de Telegram directamente (sin librerías externas pesadas).
    """
    
    def __init__(self):
        """Inicializa el notificador de Telegram"""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
    
    def send_message(self, text, silent=False):
        """
        Envía un mensaje a Telegram.
        
        Args:
            text: Texto del mensaje (soporta HTML)
            silent: Si es True, la notificación es silenciosa
        
        Returns:
            bool: True si se envió correctamente
        """
        if not self.enabled:
            return False
            
        try:
            response = requests.post(
                self.api_url,
                json={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': 'HTML',
                    'disable_notification': silent
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Error enviando mensaje a Telegram: {e}")
            return False
    
    def notify_startup(self, mode, symbols, capital):
        """
        Notificación de inicio del bot.
        
        Args:
            mode: Modo de trading ('paper' o 'live')
            symbols: Lista de pares a operar
            capital: Capital total
        """
        emoji = "🚀" if mode == "paper" else "⚡"
        mode_text = "SIMULACIÓN" if mode == "paper" else "DINERO REAL"
        
        text = f"""{emoji} <b>Bot de Trading Iniciado</b>

📊 Modo: <b>{mode_text}</b>
💰 Capital Total: <b>{capital:.2f} EUR</b>
🪙 Pares: {', '.join([s.replace('/USDT', '') for s in symbols])}
⏰ Timeframe: 4h

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        self.send_message(text)
    
    def notify_buy(self, symbol, price, qty, cost, sl_price, tp_price):
        """
        Notificación de compra.
        
        Args:
            symbol: Par (ej: 'ETH/USDT')
            price: Precio de compra
            qty: Cantidad comprada
            cost: Costo total
            sl_price: Precio de stop loss
            tp_price: Precio de take profit estimado
        """
        # Calcular potenciales
        potential_loss = ((sl_price - price) / price) * 100
        potential_gain = ((tp_price - price) / price) * 100
        
        text = f"""📈 <b>COMPRA EJECUTADA</b>

🪙 Par: <b>{symbol.replace('/USDT', '')}</b>
💵 Precio: <b>${price:.4f}</b>
📦 Cantidad: {qty:.6f}
💰 Invertido: ${cost:.2f}

🛑 Stop Loss: ${sl_price:.4f} ({potential_loss:.1f}%)
🎯 Take Profit: ${tp_price:.4f} (+{potential_gain:.1f}%)

⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        self.send_message(text)
    
    def notify_sell(self, symbol, price, qty, reason, pnl, roi):
        """
        Notificación de venta.
        
        Args:
            symbol: Par (ej: 'ETH/USDT')
            price: Precio de venta
            qty: Cantidad vendida
            reason: Razón de la venta ('TP', 'SL', 'MA_SL', 'bearish')
            pnl: Profit & Loss en USD
            roi: Retorno sobre inversión en %
        """
        emoji_map = {
            'TP': '💰',
            'SL': '🛑',
            'MA_SL': '⚠️',
            'bearish': '📉'
        }
        
        reason_map = {
            'TP': 'Take Profit',
            'SL': 'Stop Loss',
            'MA_SL': 'Stop Loss (MA)',
            'bearish': 'Señal Bajista'
        }
        
        emoji = emoji_map.get(reason, '📉')
        reason_text = reason_map.get(reason, reason)
        profit = pnl > 0
        pnl_emoji = '💚' if profit else '💔'
        
        text = f"""{emoji} <b>VENTA EJECUTADA</b>

🪙 Par: <b>{symbol.replace('/USDT', '')}</b>
💵 Precio: <b>${price:.4f}</b>
📦 Cantidad: {qty:.6f}
📋 Razón: {reason_text}

{pnl_emoji} P&L: <b>${pnl:.2f}</b> ({roi:+.2f}%)

⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        self.send_message(text)
    
    def notify_cycle_complete(self, total_equity, initial_capital, roi, positions_count):
        """
        Notificación de ciclo completado.
        
        Args:
            total_equity: Equity total actual
            initial_capital: Capital inicial
            roi: ROI total en %
            positions_count: Número de posiciones abiertas
        """
        profit = roi > 0
        emoji = '📊' if roi >= 0 else '📉'
        
        text = f"""{emoji} <b>Ciclo Completado</b>

💰 Equity: <b>${total_equity:.2f}</b>
📈 ROI Total: <b>{roi:+.2f}%</b>
{'💚' if profit else '💔'} P&L: ${total_equity - initial_capital:.2f}
📍 Posiciones: {positions_count}/4

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        self.send_message(text, silent=True)
    
    def notify_error(self, error_msg):
        """
        Notificación de error crítico.
        
        Args:
            error_msg: Descripción del error
        """
        text = f"""❌ <b>ERROR CRÍTICO</b>

{error_msg}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ Revisa los logs del bot"""
        
        self.send_message(text)
    
    def notify_update(self, old_version, new_version):
        """
        Notificación de actualización aplicada.
        
        Args:
            old_version: Versión anterior
            new_version: Nueva versión
        """
        text = f"""🔄 <b>Bot Actualizado</b>

📦 v{old_version} → v{new_version}

✅ Actualización aplicada correctamente
🔄 Bot reiniciado

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        self.send_message(text)
