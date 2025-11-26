from flask import Flask, render_template, jsonify
import json
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import os
from data_cache import DataCache

app = Flask(__name__)

# Configuración
SYMBOLS = ['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']
ATR_LENGTH = 14
MA_LENGTH = 50
LONG_MA_LENGTH = 200

# Data cache
data_cache = DataCache()


def calculate_indicators(df):
    """Calcula indicadores técnicos para el gráfico (mismo lógica que bot)"""
    df = df.copy()
    
    # ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low'] - df['prev_close']).abs()
    ], axis=1).max(axis=1)
    
    # ATR Rolling (Simple approximation for dashboard speed)
    df['atr'] = df['tr'].rolling(window=ATR_LENGTH).mean()
    
    # MAs
    df['ma'] = df['close'].rolling(window=MA_LENGTH).mean()
    df['long_ma'] = df['close'].rolling(window=LONG_MA_LENGTH).mean()
    
    # ADX (Standard Welles Wilder - Matching TradingView)
    # TradingView uses RMA (Wilder's Smoothing) which is equivalent to EMA with alpha=1/length
    
    df['dm_plus'] = (df['high'] - df['high'].shift(1)).where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 0)
    df['dm_minus'] = (df['low'].shift(1) - df['low']).where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 0)
    
    # TR smoothed for ADX (RMA)
    # alpha = 1 / length
    alpha = 1 / 14
    df['tr_sm'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    
    # Directional Indicators (smoothed with RMA)
    df['dm_plus_sm'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
    df['dm_minus_sm'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()
    
    df['di_plus'] = (df['dm_plus_sm'] / df['tr_sm']) * 100
    df['di_minus'] = (df['dm_minus_sm'] / df['tr_sm']) * 100
    
    # DX
    df['dx'] = (abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
    
    # ADX (smoothed DX with RMA)
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    # EMAs for EMA Strategy
    EMA_FAST = 15
    EMA_SLOW = 30
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    return df


# ============================================================================
# DUAL BOT HELPER FUNCTIONS
# ============================================================================

def load_bot_state(bot_name):
    """
    Carga el estado de un bot específico
    
    Args:
        bot_name: 'adx' o 'ema'
    
    Returns:
        dict con el estado del bot o None si no existe
    """
    filename = 'bot_state.json' if bot_name == 'adx' else 'bot_state_ema.json'
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    return None


def load_bot_trades(bot_name):
    """
    Carga los trades de un bot específico
    
    Args:
        bot_name: 'adx' o 'ema'
    
    Returns:
        DataFrame con los trades o DataFrame vacío
    """
    filename = 'trades_production.csv' if bot_name == 'adx' else 'trades_ema.csv'
    
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def calculate_combined_metrics(adx_state, ema_state):
    """
    Calcula métricas combinadas de ambos bots
    
    Args:
        adx_state: Estado del bot ADX
        ema_state: Estado del bot EMA
    
    Returns:
        dict con métricas combinadas
    """
    # Función helper para calcular equity real (cash + posiciones)
    def get_real_equity(state, bot_type):
        if not state:
            return 0
        
        # Equity en efectivo
        if bot_type == 'adx':
            cash_equity = state.get('total_equity', 0)
        else:  # ema
            equity_dict = state.get('equity', {})
            cash_equity = sum(equity_dict.values()) if isinstance(equity_dict, dict) else 0
        
        # Valor de posiciones abiertas
        positions = state.get('positions', {})
        position_value = 0
        
        for symbol, pos in positions.items():
            if pos and isinstance(pos, dict):
                qty = pos.get('qty', 0) or pos.get('size', 0)
                if qty > 0:
                    # Obtener precio actual del símbolo
                    try:
                        df = data_cache.get_data(symbol, '4h')
                        if df is not None and len(df) > 0:
                            current_price = df.iloc[-1]['close']
                            position_value += qty * current_price
                    except Exception as e:
                        print(f"Error getting price for {symbol}: {e}")
        
        return cash_equity + position_value
    
    # Calcular equity real de cada bot
    adx_equity = get_real_equity(adx_state, 'adx')
    ema_equity = get_real_equity(ema_state, 'ema')
    
    total_equity = adx_equity + ema_equity
    
    # Capital inicial (ajustar según configuración)
    initial_capital = 200.0  # 100 EUR por bot
    
    # ROI
    combined_roi = ((total_equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
    
    # Posiciones
    adx_positions = sum(1 for pos in adx_state.get('positions', {}).values() if pos) if adx_state else 0
    ema_positions = sum(1 for pos in ema_state.get('positions', {}).values() if pos) if ema_state else 0
    
    return {
        'total_equity': total_equity,
        'adx_equity': adx_equity,
        'ema_equity': ema_equity,
        'combined_roi': combined_roi,
        'total_positions': adx_positions + ema_positions,
        'adx_positions': adx_positions,
        'ema_positions': ema_positions,
        'adx_percentage': (adx_equity / total_equity * 100) if total_equity > 0 else 0,
        'ema_percentage': (ema_equity / total_equity * 100) if total_equity > 0 else 0
    }




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
        symbol_full = f"{symbol}/USDT"
        
        # Usar caché en lugar de API directa
        df = data_cache.get_data(symbol_full, '4h')
        
        if df is None or len(df) == 0:
            return jsonify({'error': 'No data available'}), 404
        
        # Limitar a últimas 500 velas para el gráfico
        df = df.tail(500).copy()
        
        # NUEVO: Obtener vela actual (en progreso) desde Binance API
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            current_ohlcv = exchange.fetch_ohlcv(symbol_full, '4h', limit=1)
            
            if current_ohlcv and len(current_ohlcv) > 0:
                current_candle = current_ohlcv[0]
                current_timestamp = pd.to_datetime(current_candle[0], unit='ms')
                
                # Verificar si esta vela ya está en el caché (cerrada)
                if current_timestamp not in df['timestamp'].values:
                    # Es una vela nueva en progreso, añadirla
                    current_df = pd.DataFrame([{
                        'timestamp': current_timestamp,
                        'open': current_candle[1],
                        'high': current_candle[2],
                        'low': current_candle[3],
                        'close': current_candle[4],
                        'volume': current_candle[5],
                        'is_current': True  # Flag para identificarla en el frontend
                    }])
                    
                    # Append la vela actual
                    df = pd.concat([df, current_df], ignore_index=True)
                else:
                    # La vela ya existe en caché, actualizarla con valores actuales
                    idx = df[df['timestamp'] == current_timestamp].index[0]
                    df.loc[idx, 'open'] = current_candle[1]
                    df.loc[idx, 'high'] = current_candle[2]
                    df.loc[idx, 'low'] = current_candle[3]
                    df.loc[idx, 'close'] = current_candle[4]
                    df.loc[idx, 'volume'] = current_candle[5]
                    df.loc[idx, 'is_current'] = True
        except Exception as e:
            print(f"⚠️ No se pudo obtener vela actual para {symbol_full}: {e}")
            # Continuar sin vela actual
        
        # Calcular indicadores
        df = calculate_indicators(df)
        df = df.fillna(0)
        
        # Asegurar que is_current existe en todas las filas
        if 'is_current' not in df.columns:
            df['is_current'] = False
        else:
            df['is_current'] = df['is_current'].fillna(False)
        
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




# ============================================================================
# DUAL BOT API ENDPOINTS
# ============================================================================

@app.route('/api/dual_status')
def api_dual_status():
    """Estado combinado de ambos bots"""
    try:
        adx_state = load_bot_state('adx')
        ema_state = load_bot_state('ema')
        
        combined = calculate_combined_metrics(adx_state, ema_state)
        
        return jsonify({
            'combined': combined,
            'adx': {
                'active': adx_state is not None,
                'equity': combined['adx_equity'],
                'positions': combined['adx_positions'],
                'timestamp': adx_state.get('timestamp') if adx_state else None
            },
            'ema': {
                'active': ema_state is not None,
                'equity': combined['ema_equity'],
                'positions': combined['ema_positions'],
                'timestamp': ema_state.get('last_update') if ema_state else None
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bot/<bot_name>/status')
def api_bot_status(bot_name):
    """Estado individual de un bot específico"""
    try:
        if bot_name not in ['adx', 'ema']:
            return jsonify({'error': 'Invalid bot name. Use "adx" or "ema"'}), 400
        
        state = load_bot_state(bot_name)
        
        if not state:
            return jsonify({'error': f'No state file found for bot {bot_name}'}), 404
        
        # Calcular métricas según el bot
        if bot_name == 'adx':
            total_equity = state.get('total_equity', 0)
            positions = state.get('positions', {})
        else:  # ema
            equity_dict = state.get('equity', {})
            total_equity = sum(equity_dict.values()) if isinstance(equity_dict, dict) else 0
            positions = state.get('positions', {})
        
        open_positions = sum(1 for pos in positions.values() if pos)
        
        # Capital inicial por bot
        initial_capital = 100.0
        roi = ((total_equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0
        
        return jsonify({
            'bot_name': bot_name,
            'total_equity': total_equity,
            'roi': roi,
            'open_positions': open_positions,
            'total_pairs': 4,
            'mode': state.get('mode', 'paper'),
            'timestamp': state.get('timestamp') or state.get('last_update'),
            'positions': positions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bot/<bot_name>/trades')
def api_bot_trades(bot_name):
    """Trades de un bot específico"""
    try:
        if bot_name not in ['adx', 'ema']:
            return jsonify({'error': 'Invalid bot name. Use "adx" or "ema"'}), 400
        
        df = load_bot_trades(bot_name)
        
        if df.empty:
            return jsonify([])
        
        # Últimos 20 trades
        df_recent = df.tail(20).copy()
        trades_list = df_recent.to_dict('records')
        
        return jsonify(trades_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/comparison')
def api_comparison():
    """Datos de comparación entre ambos bots"""
    try:
        adx_trades = load_bot_trades('adx')
        ema_trades = load_bot_trades('ema')
        
        adx_state = load_bot_state('adx')
        ema_state = load_bot_state('ema')
        
        # Calcular métricas de trades
        def calculate_trade_metrics(df):
            if df.empty:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            sells = df[df['type'] == 'sell']
            if sells.empty:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            wins = len(sells[sells['pnl'] > 0])
            losses = len(sells[sells['pnl'] <= 0])
            total = len(sells)
            
            return {
                'total_trades': total,
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl': sells['pnl'].sum(),
                'avg_pnl': sells['pnl'].mean()
            }
        
        adx_metrics = calculate_trade_metrics(adx_trades)
        ema_metrics = calculate_trade_metrics(ema_trades)
        
        # Equity y ROI
        adx_equity = adx_state.get('total_equity', 100) if adx_state else 100
        ema_equity_dict = ema_state.get('equity', {}) if ema_state else {}
        ema_equity = sum(ema_equity_dict.values()) if isinstance(ema_equity_dict, dict) else 100
        
        adx_roi = ((adx_equity - 100) / 100 * 100)
        ema_roi = ((ema_equity - 100) / 100 * 100)
        
        return jsonify({
            'adx': {
                'equity': adx_equity,
                'roi': adx_roi,
                **adx_metrics
            },
            'ema': {
                'equity': ema_equity,
                'roi': ema_roi,
                **ema_metrics
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# OPTIMIZER ENDPOINTS
# ============================================================================

def save_optimizer_results(results, strategy):
    """Guarda resultados de optimización en JSON"""
    try:
        filename = f'optimizer_results_{strategy}.json'
        results_dict = results.to_dict('records') if hasattr(results, 'to_dict') else results
        
        with open(filename, 'w') as f:
            json.dump({
                'strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                'results': results_dict
            }, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving optimizer results: {e}")
        return False


def load_optimizer_results(strategy):
    """Carga últimos resultados de optimización"""
    try:
        filename = f'optimizer_results_{strategy}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading optimizer results: {e}")
        return None


def run_optimizer(strategy, symbols):
    """
    Ejecuta optimización de estrategia
    
    Args:
        strategy: 'ema' o 'momentum'
        symbols: lista de símbolos a optimizar
    
    Returns:
        DataFrame con resultados ordenados por score
    """
    from strategy_optimizer import StrategyOptimizer
    
    optimizer = StrategyOptimizer()
    
    if strategy == 'ema':
        results_df = optimizer.optimize_ema_strategy(symbols)
    elif strategy == 'momentum':
        results_df = optimizer.optimize_momentum_strategy(symbols)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return results_df


@app.route('/api/optimizer/run', methods=['POST'])
def api_optimizer_run():
    """Ejecuta optimización de estrategia"""
    try:
        from flask import request
        
        data = request.get_json()
        strategy = data.get('strategy', 'ema')
        symbols = data.get('symbols', ['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT'])
        
        # Validar estrategia
        if strategy not in ['ema', 'momentum']:
            return jsonify({'error': 'Invalid strategy. Use "ema" or "momentum"'}), 400
        
        # Ejecutar optimización (esto puede tardar 2-5 minutos)
        results_df = run_optimizer(strategy, symbols)
        
        # Guardar resultados
        save_optimizer_results(results_df, strategy)
        
        # Convertir a dict para JSON
        results_dict = results_df.to_dict('records')
        
        # Top 10 por score
        top_score = results_dict[:10]
        
        # Top 10 por ROI
        results_roi = sorted(results_dict, key=lambda x: x.get('avg_roi', 0), reverse=True)
        top_roi = results_roi[:10]
        
        return jsonify({
            'success': True,
            'strategy': strategy,
            'symbols': symbols,
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(results_dict),
            'top_score': top_score,
            'top_roi': top_roi
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimizer/last-results')
def api_optimizer_last_results():
    """Obtiene últimos resultados de optimización"""
    try:
        from flask import request
        
        strategy = request.args.get('strategy', 'ema')
        
        results = load_optimizer_results(strategy)
        
        if results is None:
            return jsonify({'error': 'No results found'}), 404
        
        # Extraer top 10 por score y ROI
        all_results = results.get('results', [])
        
        top_score = all_results[:10]
        results_roi = sorted(all_results, key=lambda x: x.get('avg_roi', 0), reverse=True)
        top_roi = results_roi[:10]
        
        return jsonify({
            'success': True,
            'strategy': strategy,
            'timestamp': results.get('timestamp'),
            'total_configs': len(all_results),
            'top_score': top_score,
            'top_roi': top_roi
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

