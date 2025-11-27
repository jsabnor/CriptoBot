"""
Neural Strategy Backtesting

Backtest de la estrategia neuronal sobre datos históricos
para validar rendimiento antes de trading real.

Uso:
    python neural_backtest.py --symbol ETH/USDT --start-date 2024-01-01
    python neural_backtest.py --symbols ETH/USDT BTC/USDT SOL/USDT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse

from data_cache import DataCache
from neural_strategy import NeuralStrategy
from neural_config import config


class NeuralBacktest:
    """Backtesting para estrategia neuronal"""
    
    def __init__(self, initial_capital=200, capital_per_pair=50):
        self.initial_capital = initial_capital
        self.capital_per_pair = capital_per_pair
        self.cache = DataCache()
    
    def backtest_symbol(self, symbol, strategy, start_date=None, end_date=None):
        """
        Ejecuta backtest en un símbolo
        
        Args:
            symbol: Par de trading
            strategy: Instancia de NeuralStrategy
            start_date: Fecha inicial (str formato YYYY-MM-DD)
            end_date: Fecha final (str formato YYYY-MM-DD)
        
        Returns:
            dict con métricas de rendimiento
        """
        print(f"\n{'='*60}")
        print(f"Backtesting {symbol}")
        print(f"{'='*60}")
        
        # Cargar datos UNA SOLA VEZ
        df = self.cache.get_data(symbol, config.DEFAULT_TIMEFRAME)
        
        if df is None or len(df) < config.LOOKBACK_WINDOW:
            print(f"❌ Datos insuficientes para {symbol}")
            return None
        
        # Filtrar por fechas
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
        print(f"📊 Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
        print(f"   Velas: {len(df)}")
        
        # CRÍTICO: Extraer features UNA SOLA VEZ para todos los datos
        print(f"🔧 Extrayendo features...")
        X = strategy.feature_extractor.extract_features(df, fit_scaler=False)
        X_seq = strategy.feature_extractor.create_sequences(X)
        
        if len(X_seq) == 0:
            print(f"❌ No se pudieron crear secuencias")
            return None
        
        print(f"✅ {len(X_seq)} predicciones generadas")
        
        # Generar TODAS las predicciones de una vez (eficiente)
        print(f"🧠 Generando predicciones...")
        predictions = strategy.model.predict(X_seq)
        
        # Simular trading
        capital = self.capital_per_pair
        position = None
        trades = []
        equity_curve = []
        
        # Ajustar índices: X_seq tiene lookback menos elementos que df
        start_idx = config.LOOKBACK_WINDOW
        
        for i in range(len(predictions)):
            df_idx = start_idx + i
            
            if df_idx >= len(df):
                break
            
            current_time = df.iloc[df_idx]['timestamp']
            current_price = df.iloc[df_idx]['close']
            
            # Obtener predicción pre-calculada
            probs = predictions[i]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            
            # Aplicar umbrales
            signal = config.CLASS_LABELS[predicted_class]
            if signal == 'BUY' and confidence < config.MIN_CONFIDENCE_BUY:
                signal = 'HOLD'
            elif signal == 'SELL' and confidence < config.MIN_CONFIDENCE_SELL:
                signal = 'HOLD'
            
            # Lógica de trading
            if position is None:
                # Sin posición - buscar entrada
                if signal == 'BUY':
                    position = {
                        'entry_price': current_price,
                        'size': capital / current_price,
                        'entry_time': current_time,
                        'confidence': confidence,
                        'highest_price': current_price  # Para Trailing Stop
                    }
                    print(f"  🟢 BUY @ {current_price:.2f} ({current_time.strftime('%Y-%m-%d')}) - Conf: {confidence:.2%}")
            
            else:
                # Con posición - gestionar salida
                # Actualizar highest price para Trailing Stop
                position['highest_price'] = max(position['highest_price'], current_price)
                
                # Calcular PnL
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                
                # Calcular Trailing Drawdown desde el máximo
                trailing_drawdown = (position['highest_price'] - current_price) / position['highest_price']
                
                # Condiciones de salida
                should_close = False
                exit_reason = ''
                
                if signal == 'SELL':
                    should_close = True
                    exit_reason = 'Señal SELL'
                
                # Stop loss fijo (-4%)
                elif pnl_pct <= -0.04:
                    should_close = True
                    exit_reason = 'Stop Loss'
                
                # Trailing Stop (3%) - Solo si estamos en ganancias (>1%)
                elif pnl_pct > 0.01 and trailing_drawdown >= 0.03:
                    should_close = True
                    exit_reason = 'Trailing Stop'
                
                if should_close:
                    # Cerrar posición
                    exit_value = position['size'] * current_price
                    profit = exit_value - capital
                    profit_pct = (exit_value / capital - 1)
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'exit_reason': exit_reason,
                        'entry_confidence': position['confidence']
                    }
                    
                    trades.append(trade)
                    capital += profit
                    
                    print(f"  🔴 SELL @ {current_price:.2f} ({current_time.strftime('%Y-%m-%d')})")
                    print(f"     Profit: ${profit:.2f} ({profit_pct:.2%}) - {exit_reason}")
                    
                    position = None
            
            # Guardar equity
            equity_value = capital
            if position is not None:
                equity_value = position['size'] * current_price
            
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity_value
            })
        
        # Calcular métricas
        metrics = self.calculate_metrics(trades, equity_curve, self.capital_per_pair)
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS - {symbol}")
        print(f"{'='*60}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"ROI: {metrics['roi']:.2%}")
        print(f"Final Capital: ${metrics['final_capital']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'symbol': symbol,
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def calculate_metrics(self, trades, equity_curve, initial_capital):
        """Calcula métricas de rendimiento"""
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'final_capital': initial_capital,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }
        
        # Trades ganadores/perdedores
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # ROI
        final_capital = equity_curve[-1]['equity'] if equity_curve else initial_capital
        roi = (final_capital - initial_capital) / initial_capital
        
        # Max Drawdown
        max_equity = initial_capital
        max_drawdown = 0
        
        for point in equity_curve:
            equity = point['equity']
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
        
        # Returns para Sharpe
        equity_values = [p['equity'] for p in equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        sharpe_ratio = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 / 4)  # Anualizado para 4h
        
        # Profit/Loss promedio
        avg_profit = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'roi': roi,
            'final_capital': final_capital,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }
    
    def backtest_multiple(self, symbols, start_date=None, end_date=None):
        """Backtest en múltiples símbolos"""
        
        # Cargar estrategia
        strategy = NeuralStrategy()
        
        results = []
        
        for symbol in symbols:
            result = self.backtest_symbol(symbol, strategy, start_date, end_date)
            if result is not None:
                results.append(result)
        
        # Resumen general
        if results:
            print(f"\n{'='*60}")
            print("RESUMEN GENERAL")
            print(f"{'='*60}")
            
            total_trades = sum(r['metrics']['total_trades'] for r in results)
            avg_win_rate = np.mean([r['metrics']['win_rate'] for r in results])
            avg_roi = np.mean([r['metrics']['roi'] for r in results])
            avg_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in results])
            
            print(f"Símbolos: {len(results)}")
            print(f"Total Trades: {total_trades}")
            print(f"Avg Win Rate: {avg_win_rate:.2%}")
            print(f"Avg ROI: {avg_roi:.2%}")
            print(f"Avg Sharpe: {avg_sharpe:.2f}")
            print(f"{'='*60}\n")
            
            # Guardar resultados
            self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """Guarda resultados del backtest"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neural_backtest_results_{timestamp}.json"
        
        # Convertir timestamps a string para JSON
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            
            # Convertir trades
            trades = []
            for t in r_copy['trades']:
                t_copy = t.copy()
                t_copy['entry_time'] = t_copy['entry_time'].isoformat()
                t_copy['exit_time'] = t_copy['exit_time'].isoformat()
                trades.append(t_copy)
            r_copy['trades'] = trades
            
            # Convertir equity curve
            equity = []
            for e in r_copy['equity_curve']:
                e_copy = e.copy()
                e_copy['timestamp'] = e_copy['timestamp'].isoformat()
                equity.append(e_copy)
            r_copy['equity_curve'] = equity
            
            results_serializable.append(r_copy)
        
        path = Path('models') / 'logs' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Helper para convertir numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj
            
        results_clean = convert_numpy(results_serializable)
        
        with open(path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"💾 Resultados guardados: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Strategy Backtesting')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Símbolo único para backtest')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Múltiples símbolos para backtest')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='Fecha fin (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=50,
                       help='Capital por par')
    
    args = parser.parse_args()
    
    # Determinar símbolos
    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = config.DEFAULT_SYMBOLS
    
    print("\n" + "="*60)
    print("🧪 NEURAL STRATEGY BACKTESTING")
    print("="*60)
    print(f"Símbolos: {symbols}")
    print(f"Período: {args.start_date or 'inicio'} a {args.end_date or 'fin'}")
    print(f"Capital por par: ${args.capital}")
    print("="*60 + "\n")
    
    # Ejecutar backtest
    backtester = NeuralBacktest(capital_per_pair=args.capital)
    results = backtester.backtest_multiple(symbols, args.start_date, args.end_date)
