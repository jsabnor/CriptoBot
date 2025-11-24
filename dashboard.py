from flask import Flask, render_template, jsonify
import json
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import os

app = Flask(__name__)

# Configuración
SYMBOLS = ['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']
ATR_LENGTH = 14
MA_LENGTH = 50
LONG_MA_LENGTH = 200


def calculate_indicators(df):
    """Calcula indicadores técnicos para el gráfico"""
    # ATR
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=ATR_LENGTH).mean()
    
    # MAs
    df['ma'] = df['close'].rolling(window=MA_LENGTH).mean()
    df['long_ma'] = df['close'].rolling(window=LONG_MA_LENGTH).mean()
    
    # ADX (simplificado)
    df['dx'] = abs((df['high'] - df['high'].shift(1)) - (df['low'].shift(1) - df['low']))
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    return df


@app.route('/')
def index():
    """Página principal del dashboard"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Estado actual del bot (desde bot_state.json)"""
    try:
        if os.path.exists('bot_state.json'):
            with open('bot_state.json', 'r') as f:
                state = json.load(f)
            
            # Calcular posiciones abiertas
            open_positions = sum(1 for pos in state.get('positions', {}).values() if pos is not None)
            
            # Calcular ROI
            total_equity = state.get('total_equity', 200.0)
            initial_capital = 200.0  # 4 pares x 50 EUR
            roi = ((total_equity - initial_capital) / initial_capital) * 100
            
            return jsonify({
                'total_equity': total_equity,
                'roi': roi,
                'open_positions': open_positions,
                'total_pairs': 4,
                'mode': state.get('mode', 'paper'),
                'timestamp': state.get('timestamp', datetime.now().isoformat()),
                'equity_by_pair': state.get('equity', {}),
                'positions': state.get('positions', {})
            })
        else:
            return jsonify({
                'total_equity': 200.0,
                'roi': 0.0,
                'open_positions': 0,
                'total_pairs': 4,
                'mode': 'paper',
                'timestamp': datetime.now().isoformat(),
                'equity_by_pair': {},
                'positions': {}
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades')
def api_trades():
    """Historial de trades (desde trades_production.csv)"""
    try:
        if os.path.exists('trades_production.csv'):
            df = pd.read_csv('trades_production.csv')
            
            # Últimos 20 trades
            df_recent = df.tail(20).copy()
            
            # Formatear para JSON
            trades_list = df_recent.to_dict('records')
            
            return jsonify(trades_list)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    """Datos para gráfico de velas con indicadores y marcadores de trades"""
    try:
        # Fetch OHLCV data from Binance
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_full = f"{symbol}/USDT"
        
        ohlcv = exchange.fetch_ohlcv(symbol_full, '4h', limit=200)
        
        # Convertir a DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calcular indicadores
        df = calculate_indicators(df)
        
        # Eliminar NaN para evitar problemas en el frontend
        df = df.fillna(0)
        
        # Obtener trades de este símbolo
        trades = []
        if os.path.exists('trades_production.csv'):
            trades_df = pd.read_csv('trades_production.csv')
            symbol_trades = trades_df[trades_df['symbol'] == symbol_full]
            
            for _, trade in symbol_trades.iterrows():
                trades.append({
                    'timestamp': trade['timestamp'],
                    'type': trade['type'],
                    'price': float(trade['price']),
                    'qty': float(trade['qty']),
                    'reason': trade.get('reason', ''),
                    'pnl': float(trade.get('pnl', 0))
                })
        
        return jsonify({
            'candles': df.to_dict('records'),
            'trades': trades,
            'symbol': symbol
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    # Ejecutar Flask en modo desarrollo
    # En producción, usar gunicorn o similar
    app.run(host='0.0.0.0', port=5000, debug=False)
