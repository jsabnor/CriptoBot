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
    
    def send_message(self, text, silent=False, buttons=None):
        """
        Envía un mensaje a Telegram.
        
        Args:
            text: Texto del mensaje (soporta HTML)
            silent: Si es True, la notificación es silenciosa
            buttons: Lista de botones inline [[{'text': 'Label', 'url': 'URL'}]]
        
        Returns:
            bool: True si se envió correctamente
        """
        if not self.enabled:
            return False
            
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
                'disable_notification': silent
            }
            
            # Añadir botones si se proporcionan
            if buttons:
                payload['reply_markup'] = {'inline_keyboard': buttons}
            
            response = requests.post(
                self.api_url,
                json=payload,
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
    
    def notify_buy(self, symbol, price, qty, cost, sl_price, tp_price, adx=None, ma_status=None, strategy_name=''):
        """
        Notificación de compra mejorada.
        
        Args:
            symbol: Par (ej: 'ETH/USDT')
            price: Precio de compra
            qty: Cantidad comprada
            cost: Costo total
            sl_price: Precio de stop loss
            tp_price: Precio de take profit estimado
            adx: Valor del ADX (opcional)
            ma_status: Estado de MA ('bullish' o 'bearish', opcional)
            strategy_name: Nombre de la estrategia (opcional, ej: 'ADX', 'EMA')
        """
        # Calcular potenciales
        potential_loss = ((sl_price - price) / price) * 100
        potential_gain = ((tp_price - price) / price) * 100
        risk_reward = abs(potential_gain / potential_loss) if potential_loss != 0 else 0
        
        # Emojis contextuales
        adx_emoji = '🔥' if adx and adx > 30 else '✅' if adx and adx > 25 else '⚡'
        ma_emoji = '✅' if ma_status == 'bullish' else '⚠️' if ma_status else '➖'
        
        # Prefijo de estrategia
        strategy_prefix = f"[{strategy_name}] " if strategy_name else ""
        
        # Construir mensaje
        symbol_clean = symbol.replace('/USDT', '')
        text = f"""🟢 <b>{strategy_prefix}COMPRA EJECUTADA</b>

━━━━━━━━━━━━━━━━━━━━
🪙 <b>{symbol_clean}/USDT</b>
━━━━━━━━━━━━━━━━━━━━

💵 Precio: <b>${price:.4f}</b>
📦 Cantidad: <b>{qty:.6f}</b>
💰 Invertido: <b>${cost:.2f}</b>"""

        # Añadir indicadores si están disponibles
        if adx is not None or ma_status is not None:
            text += f"""

📊 <b>Indicadores:</b>"""
            if adx is not None:
                trend_text = 'Muy Fuerte' if adx > 30 else 'Fuerte' if adx > 25 else 'Moderada'
                text += f"""
├─ ADX: {adx:.1f} {adx_emoji}
├─ Tendencia: {trend_text}"""
            if ma_status is not None:
                ma_text = 'Alcista' if ma_status == 'bullish' else 'Bajista'
                text += f"""
└─ MA50: {ma_emoji} {ma_text}"""
        
        # Objetivos
        text += f"""

🎯 <b>Objetivos:</b>
├─ 🛑 SL: ${sl_price:.4f} ({potential_loss:.1f}%)
├─ 💰 TP: ${tp_price:.4f} (+{potential_gain:.1f}%)
└─ ⚖️ R:R = 1:{risk_reward:.1f}

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        # Botones interactivos
        dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:5000')
        tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_clean}USDT"
        
        buttons = [[
            {'text': '📊 Ver Dashboard', 'url': dashboard_url},
            {'text': '📈 TradingView', 'url': tradingview_url}
        ]]
        
        self.send_message(text, buttons=buttons)
    
    def notify_sell(self, symbol, price, qty, reason, pnl, roi, entry_price=None, duration=None, strategy_name=''):
        """
        Notificación de venta mejorada.
        
        Args:
            symbol: Par (ej: 'ETH/USDT')
            price: Precio de venta
            qty: Cantidad vendida
            reason: Razón de la venta ('TP', 'SL', 'MA_SL', 'bearish')
            pnl: Profit & Loss en USD
            roi: Retorno sobre inversión en %
            entry_price: Precio de entrada (opcional)
            duration: Duración del trade en formato string (opcional)
            strategy_name: Nombre de la estrategia (opcional)
        """
        emoji_map = {
            'TP': '💰',
            'SL': '🛑',
            'MA_SL': '⚠️',
            'bearish': '📉',
            'Signal': '📊'
        }
        
        reason_map = {
            'TP': 'Take Profit',
            'SL': 'Stop Loss',
            'MA_SL': 'Stop Loss (MA)',
            'bearish': 'Señal Bajista',
            'Signal': 'Señal de Salida'
        }
        
        emoji = emoji_map.get(reason, '📉')
        reason_text = reason_map.get(reason, reason)
        profit = pnl > 0
        result_emoji = '🟢' if profit else '🔴'
        pnl_emoji = '💚' if profit else '💔'
        
        # Prefijo de estrategia
        strategy_prefix = f"[{strategy_name}] " if strategy_name else ""
        
        symbol_clean = symbol.replace('/USDT', '')
        text = f"""{result_emoji} <b>{strategy_prefix}VENTA EJECUTADA</b>

━━━━━━━━━━━━━━━━━━━━
🪙 <b>{symbol_clean}/USDT</b>
━━━━━━━━━━━━━━━━━━━━

📊 <b>Operación:</b>"""

        if entry_price:
            text += f"""
├─ Entrada: ${entry_price:.4f}
├─ Salida: ${price:.4f}"""
        else:
            text += f"""
├─ Precio: ${price:.4f}"""
        
        text += f"""
├─ Cantidad: {qty:.6f}"""
        
        if duration:
            text += f"""
└─ Duración: {duration}"""
        else:
            text += f"""
└─ Razón: {reason_text}"""
        
        text += f"""

💰 <b>Resultado:</b>
├─ P&L: <b>${pnl:+.2f}</b>
├─ ROI: <b>{roi:+.2f}%</b>
└─ {emoji} {reason_text}

{pnl_emoji} {'¡Ganancia!' if profit else 'Pérdida'}

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        # Botones interactivos
        dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:5000')
        
        buttons = [[
            {'text': '📊 Ver Dashboard', 'url': dashboard_url},
            {'text': '📋 Ver Historial', 'url': f'{dashboard_url}#trades'}
        ]]
        
        self.send_message(text, buttons=buttons)
    
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
    
    def notify_milestone(self, milestone_type, value, details=''):
        """
        Notificación de milestone alcanzado.
        
        Args:
            milestone_type: Tipo de milestone ('roi_10', 'roi_20', 'trades_100', etc.)
            value: Valor del milestone
            details: Detalles adicionales (opcional)
        """
        milestones = {
            'roi_10': ('🎉', 'ROI +10% Alcanzado!'),
            'roi_20': ('🚀', 'ROI +20% Alcanzado!'),
            'roi_50': ('💎', 'ROI +50% Alcanzado!'),
            'trades_50': ('🎯', '50 Trades Completados!'),
            'trades_100': ('💯', '100 Trades Completados!'),
            'trades_200': ('🏆', '200 Trades Completados!'),
            'win_streak_5': ('🔥', 'Racha de 5 Wins!'),
            'win_streak_10': ('🔥🔥', 'Racha de 10 Wins!'),
        }
        
        emoji, title = milestones.get(milestone_type, ('🎊', 'Milestone Alcanzado'))
        
        text = f"""{emoji} <b>{title}</b>

🏆 Has alcanzado un nuevo hito

📊 Valor: {value}"""
        
        if details:
            text += f"""
ℹ️ {details}"""
        
        text += f"""

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

¡Sigue así! 💪"""
        
        self.send_message(text)
    
    def notify_risk_alert(self, alert_type, details):
        """
        Notificación de alerta de riesgo.
        
        Args:
            alert_type: Tipo de alerta
            details: Detalles de la alerta
        """
        text = f"""⚠️ <b>ALERTA DE RIESGO</b>

🚨 {alert_type}

{details}

⚠️ Revisa el dashboard inmediatamente

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:5000')
        buttons = [[
            {'text': '📊 Ver Dashboard', 'url': dashboard_url}
        ]]
        
        self.send_message(text, buttons=buttons)
    
    def notify_strong_signal(self, symbol, adx, price, reason=''):
        """
        Notificación de señal fuerte detectada sin posición abierta.
        
        Args:
            symbol: Par de trading
            adx: Valor del ADX
            price: Precio actual
            reason: Razón adicional (opcional)
        """
        symbol_clean = symbol.replace('/USDT', '')
        
        text = f"""⚡ <b>SEÑAL FUERTE DETECTADA</b>

🪙 Par: <b>{symbol_clean}/USDT</b>
📊 ADX: <b>{adx:.1f}</b> {'🔥' if adx > 30 else '✅'}
💵 Precio Actual: ${price:.4f}

{'📋 ' + reason if reason else ''}

⚠️ Sin posición abierta actualmente

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        tradingview_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_clean}USDT"
        buttons = [[
            {'text': '📈 Ver en TradingView', 'url': tradingview_url}
        ]]
        
        self.send_message(text, buttons=buttons)
    
    def notify_daily_summary(self, stats):
        """
        Resumen diario de trading.
        
        Args:
            stats: Diccionario con estadísticas del día
        """
        profit = stats.get('pnl', 0) > 0
        emoji = '📈' if profit else '📉'
        
        text = f"""{emoji} <b>Resumen Diario</b>

🗓️ {datetime.now().strftime('%d de %B de %Y')}

━━━━━━━━━━━━━━━━━━━━
📊 <b>Performance</b>
━━━━━━━━━━━━━━━━━━━━

💰 P&L del Día: <b>${stats.get('pnl', 0):+.2f}</b>
📈 ROI del Día: <b>{stats.get('roi', 0):+.2f}%</b>
🎯 Trades: {stats.get('total_trades', 0)} ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)
✅ Win Rate: {stats.get('win_rate', 0):.1f}%

━━━━━━━━━━━━━━━━━━━━
💼 <b>Estado Actual</b>
━━━━━━━━━━━━━━━━━━━━

💰 Equity Total: ${stats.get('total_equity', 0):.2f}
📈 ROI Acumulado: {stats.get('total_roi', 0):+.2f}%
📍 Posiciones Abiertas: {stats.get('open_positions', 0)}/4

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:5000')
        buttons = [[
            {'text': '📊 Ver Dashboard', 'url': dashboard_url}
        ]]
        
        self.send_message(text, silent=True, buttons=buttons)
    
    def notify_weekly_summary(self, stats):
        """
        Resumen semanal de trading.
        
        Args:
            stats: Diccionario con estadísticas de la semana
        """
        profit = stats.get('pnl', 0) > 0
        emoji = '📊' if profit else '📉'
        
        text = f"""{emoji} <b>Resumen Semanal</b>

🗓️ Semana del {stats.get('week_start', '')} al {stats.get('week_end', '')}

━━━━━━━━━━━━━━━━━━━━
📈 <b>Performance</b>
━━━━━━━━━━━━━━━━━━━━

💰 P&L Semanal: <b>${stats.get('pnl', 0):+.2f}</b>
📊 ROI Semanal: <b>{stats.get('roi', 0):+.2f}%</b>
🎯 Trades: {stats.get('total_trades', 0)} ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)
✅ Win Rate: {stats.get('win_rate', 0):.1f}%

━━━━━━━━━━━━━━━━━━━━
🏆 <b>Top Performers</b>
━━━━━━━━━━━━━━━━━━━━

1️⃣ {stats.get('best_pair', 'N/A')}: {stats.get('best_roi', 0):+.2f}%
2️⃣ {stats.get('second_pair', 'N/A')}: {stats.get('second_roi', 0):+.2f}%

━━━━━━━━━━━━━━━━━━━━
💼 <b>Estado Actual</b>
━━━━━━━━━━━━━━━━━━━━

💰 Equity Total: ${stats.get('total_equity', 0):.2f}
📈 ROI Total: {stats.get('total_roi', 0):+.2f}%
📍 Posiciones Abiertas: {stats.get('open_positions', 0)}/4

⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"""
        
        dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:5000')
        buttons = [[
            {'text': '📊 Ver Dashboard', 'url': dashboard_url}
        ]]
        
        self.send_message(text, silent=True, buttons=buttons)

