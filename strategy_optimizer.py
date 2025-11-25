"""
Strategy Optimizer - Optimiza EMA Crossover y Momentum

Busca los mejores parámetros para:
- Maximizar ROI
- Aumentar Win Rate
- Reducir Drawdown

Uso:
    python strategy_optimizer.py --strategy ema
    python strategy_optimizer.py --strategy momentum
    python strategy_optimizer.py --strategy both
"""

import pandas as pd
import numpy as np
from datetime import datetime
from data_cache import DataCache
import warnings
warnings.filterwarnings('ignore')
from itertools import product

class StrategyOptimizer:
    """Optimizador de estrategias EMA y Momentum"""
    
    def __init__(self, initial_capital=200, capital_per_pair=50):
        self.initial_capital = initial_capital
        self.capital_per_pair = capital_per_pair
        self.commission = 0.001
        self.cache = DataCache()
        
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
    
    def calculate_rsi(self, df, period=14):
        """Calcula RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ==================== ESTRATEGIAS OPTIMIZADAS ====================
    
    def strategy_ema_optimized(self, df, fast=12, slow=26, use_filter=True, 
                               rsi_filter=False, atr_sl=False, atr_mult=2.0):
        """
        EMA Crossover Optimizada
        
        Parámetros:
        - fast: Período EMA rápida
        - slow: Período EMA lenta
        - use_filter: Usar filtro de tendencia (EMA 200)
        - rsi_filter: Filtrar con RSI
        - atr_sl: Usar ATR para stop loss
        - atr_mult: Multiplicador ATR
        """
        df = df.copy()
        df['ema_fast'] = self.calculate_ema(df, fast)
        df['ema_slow'] = self.calculate_ema(df, slow)
        
        # Filtro de tendencia opcional
        if use_filter:
            df['ema_trend'] = self.calculate_ema(df, 200)
            trend_filter = df['close'] > df['ema_trend']
        else:
            trend_filter = True
        
        # Filtro RSI opcional
        if rsi_filter:
            df['rsi'] = self.calculate_rsi(df, 14)
            rsi_buy_filter = (df['rsi'] > 30) & (df['rsi'] < 70)
        else:
            rsi_buy_filter = True
        
        # Señales de compra
        df['buy_signal'] = (
            (df['ema_fast'] > df['ema_slow']) & 
            (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &
            trend_filter &
            rsi_buy_filter
        )
        
        # Señales de venta
        df['sell_signal'] = (
            (df['ema_fast'] < df['ema_slow']) & 
            (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        )
        
        # Stop loss ATR opcional
        if atr_sl:
            df['atr'] = self.calculate_atr(df, 14)
            df['atr_sl'] = df['close'] - (df['atr'] * atr_mult)
        
        return df
    
    def strategy_momentum_optimized(self, df, period=14, ma_period=5, 
                                    use_trend_filter=True, rsi_filter=False,
                                    min_roc=0, atr_sl=False, atr_mult=2.0):
        """
        Momentum Optimizada
        
        Parámetros:
        - period: Período ROC
        - ma_period: Período MA del ROC
        - use_trend_filter: Usar filtro de tendencia
        - rsi_filter: Filtrar con RSI
        - min_roc: ROC mínimo para comprar
        - atr_sl: Usar ATR para stop loss
        - atr_mult: Multiplicador ATR
        """
        df = df.copy()
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        df['roc_ma'] = df['roc'].rolling(window=ma_period).mean()
        
        # Filtro de tendencia opcional
        if use_trend_filter:
            df['ema_trend'] = self.calculate_ema(df, 50)
            trend_filter = df['close'] > df['ema_trend']
        else:
            trend_filter = True
        
        # Filtro RSI opcional
        if rsi_filter:
            df['rsi'] = self.calculate_rsi(df, 14)
            rsi_buy_filter = (df['rsi'] > 30) & (df['rsi'] < 70)
        else:
            rsi_buy_filter = True
        
        # Señales de compra
        df['buy_signal'] = (
            (df['roc'] > min_roc) & 
            (df['roc'] > df['roc_ma']) &
            (df['roc'].shift(1) <= df['roc_ma'].shift(1)) &
            trend_filter &
            rsi_buy_filter
        )
        
        # Señales de venta
        df['sell_signal'] = (df['roc'] < 0)
        
        # Stop loss ATR opcional
        if atr_sl:
            df['atr'] = self.calculate_atr(df, 14)
            df['atr_sl'] = df['close'] - (df['atr'] * atr_mult)
        
        return df
    
    # ==================== BACKTESTING ====================
    
    def backtest_strategy(self, symbol, df, use_atr_sl=False):
        """Ejecuta backtest con gestión de riesgo mejorada"""
        capital = self.capital_per_pair
        position = None
        trades = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Compra
            if row['buy_signal'] and position is None:
                entry_price = row['close']
                
                # Calcular stop loss
                if use_atr_sl and 'atr_sl' in df.columns:
                    sl_price = row['atr_sl']
                else:
                    sl_price = entry_price * 0.95  # SL por defecto 5%
                
                # Calcular tamaño de posición basado en riesgo
                risk_per_trade = capital * 0.02  # 2% de riesgo
                risk_per_unit = entry_price - sl_price
                
                if risk_per_unit > 0:
                    qty = risk_per_trade / risk_per_unit
                    cost = qty * entry_price * (1 + self.commission)
                    
                    if cost <= capital:
                        position = {
                            'entry_price': entry_price,
                            'sl_price': sl_price,
                            'qty': qty,
                            'entry_date': row['timestamp']
                        }
                        capital -= cost
            
            # Gestión de posición abierta
            elif position is not None:
                # Check stop loss
                if row['low'] <= position['sl_price']:
                    exit_price = position['sl_price']
                    proceeds = position['qty'] * exit_price * (1 - self.commission)
                    pnl = proceeds - (position['qty'] * position['entry_price'])
                    roi = (pnl / (position['qty'] * position['entry_price'])) * 100
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': row['timestamp'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'roi': roi,
                        'exit_reason': 'SL'
                    })
                    
                    capital += proceeds
                    position = None
                
                # Check señal de venta
                elif row['sell_signal']:
                    exit_price = row['close']
                    proceeds = position['qty'] * exit_price * (1 - self.commission)
                    pnl = proceeds - (position['qty'] * position['entry_price'])
                    roi = (pnl / (position['qty'] * position['entry_price'])) * 100
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': row['timestamp'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'pnl': pnl,
                        'roi': roi,
                        'exit_reason': 'Signal'
                    })
                    
                    capital += proceeds
                    position = None
        
        # Cerrar posición abierta
        if position is not None:
            exit_price = df.iloc[-1]['close']
            proceeds = position['qty'] * exit_price * (1 - self.commission)
            pnl = proceeds - (position['qty'] * position['entry_price'])
            roi = (pnl / (position['qty'] * position['entry_price'])) * 100
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.iloc[-1]['timestamp'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'qty': position['qty'],
                'pnl': pnl,
                'roi': roi,
                'exit_reason': 'Close'
            })
            
            capital += proceeds
        
        # Calcular métricas
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        total_roi = ((capital - self.capital_per_pair) / self.capital_per_pair) * 100
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (wins / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        # Calcular drawdown
        equity_curve = [self.capital_per_pair]
        for _, trade in trades_df.iterrows():
            equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        peak = equity_curve[0]
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = ((equity - peak) / peak) * 100
            if dd < max_dd:
                max_dd = dd
        
        # Calcular Sharpe ratio simplificado
        returns = trades_df['roi'].values
        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_roi': total_roi,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'avg_trade_roi': trades_df['roi'].mean(),
            'final_capital': capital
        }
    
    def optimize_ema_strategy(self, symbols=['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']):
        """Optimiza estrategia EMA Crossover"""
        print("=" * 80)
        print("OPTIMIZING EMA CROSSOVER STRATEGY")
        print("=" * 80)
        print()
        
        # Parámetros a probar
        fast_periods = [8, 12, 15]
        slow_periods = [21, 26, 30]
        filters = [
            {'use_filter': False, 'rsi_filter': False, 'atr_sl': False},
            {'use_filter': True, 'rsi_filter': False, 'atr_sl': False},
            {'use_filter': True, 'rsi_filter': True, 'atr_sl': False},
            {'use_filter': True, 'rsi_filter': True, 'atr_sl': True, 'atr_mult': 2.0},
            {'use_filter': True, 'rsi_filter': True, 'atr_sl': True, 'atr_mult': 3.0},
        ]
        
        results = []
        total_tests = len(fast_periods) * len(slow_periods) * len(filters) * len(symbols)
        current = 0
        
        for fast in fast_periods:
            for slow in slow_periods:
                if fast >= slow:
                    continue
                    
                for filter_config in filters:
                    config_results = []
                    
                    for symbol in symbols:
                        current += 1
                        print(f"\rTesting {current}/{total_tests}...", end='')
                        
                        df = self.cache.get_data(symbol, '4h')
                        if df is None:
                            continue
                        
                        df = df[df['timestamp'] >= '2020-01-01'].copy()
                        df = self.strategy_ema_optimized(df, fast, slow, **filter_config)
                        
                        result = self.backtest_strategy(symbol, df, filter_config.get('atr_sl', False))
                        
                        if result:
                            config_results.append(result)
                    
                    if config_results:
                        avg_roi = np.mean([r['total_roi'] for r in config_results])
                        avg_win_rate = np.mean([r['win_rate'] for r in config_results])
                        avg_dd = np.mean([r['max_drawdown'] for r in config_results])
                        avg_sharpe = np.mean([r['sharpe'] for r in config_results])
                        
                        # Score: ROI * win_rate / abs(drawdown)
                        score = (avg_roi * (avg_win_rate/100)) / abs(avg_dd) if avg_dd != 0 else 0
                        
                        results.append({
                            'fast': fast,
                            'slow': slow,
                            **filter_config,
                            'avg_roi': avg_roi,
                            'avg_win_rate': avg_win_rate,
                            'avg_drawdown': avg_dd,
                            'avg_sharpe': avg_sharpe,
                            'score': score
                        })
        
        print("\n")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n🏆 TOP 10 EMA CONFIGURATIONS (by Score)")
        print("=" * 80)
        print(results_df.head(10).to_string(index=False))
        
        print("\n\n💰 TOP 10 EMA CONFIGURATIONS (by ROI)")
        print("=" * 80)
        results_roi = results_df.sort_values('avg_roi', ascending=False)
        print(results_roi.head(10).to_string(index=False))
        
        return results_df
    
    def optimize_momentum_strategy(self, symbols=['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']):
        """Optimiza estrategia Momentum"""
        print("=" * 80)
        print("OPTIMIZING MOMENTUM STRATEGY")
        print("=" * 80)
        print()
        
        # Parámetros a probar
        periods = [10, 14, 20]
        ma_periods = [3, 5, 7]
        min_rocs = [0, 2, 5]
        filters = [
            {'use_trend_filter': False, 'rsi_filter': False, 'atr_sl': False},
            {'use_trend_filter': True, 'rsi_filter': False, 'atr_sl': False},
            {'use_trend_filter': True, 'rsi_filter': True, 'atr_sl': False},
            {'use_trend_filter': True, 'rsi_filter': True, 'atr_sl': True, 'atr_mult': 2.0},
            {'use_trend_filter': True, 'rsi_filter': True, 'atr_sl': True, 'atr_mult': 3.0},
        ]
        
        results = []
        total_tests = len(periods) * len(ma_periods) * len(min_rocs) * len(filters) * len(symbols)
        current = 0
        
        for period in periods:
            for ma_period in ma_periods:
                for min_roc in min_rocs:
                    for filter_config in filters:
                        config_results = []
                        
                        for symbol in symbols:
                            current += 1
                            print(f"\rTesting {current}/{total_tests}...", end='')
                            
                            df = self.cache.get_data(symbol, '4h')
                            if df is None:
                                continue
                            
                            df = df[df['timestamp'] >= '2020-01-01'].copy()
                            df = self.strategy_momentum_optimized(df, period, ma_period, 
                                                                 min_roc=min_roc, **filter_config)
                            
                            result = self.backtest_strategy(symbol, df, filter_config.get('atr_sl', False))
                            
                            if result:
                                config_results.append(result)
                        
                        if config_results:
                            avg_roi = np.mean([r['total_roi'] for r in config_results])
                            avg_win_rate = np.mean([r['win_rate'] for r in config_results])
                            avg_dd = np.mean([r['max_drawdown'] for r in config_results])
                            avg_sharpe = np.mean([r['sharpe'] for r in config_results])
                            
                            score = (avg_roi * (avg_win_rate/100)) / abs(avg_dd) if avg_dd != 0 else 0
                            
                            results.append({
                                'period': period,
                                'ma_period': ma_period,
                                'min_roc': min_roc,
                                **filter_config,
                                'avg_roi': avg_roi,
                                'avg_win_rate': avg_win_rate,
                                'avg_drawdown': avg_dd,
                                'avg_sharpe': avg_sharpe,
                                'score': score
                            })
        
        print("\n")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n🏆 TOP 10 MOMENTUM CONFIGURATIONS (by Score)")
        print("=" * 80)
        print(results_df.head(10).to_string(index=False))
        
        print("\n\n💰 TOP 10 MOMENTUM CONFIGURATIONS (by ROI)")
        print("=" * 80)
        results_roi = results_df.sort_values('avg_roi', ascending=False)
        print(results_roi.head(10).to_string(index=False))
        
        return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize EMA and Momentum strategies')
    parser.add_argument('--strategy', type=str, default='both', 
                       choices=['ema', 'momentum', 'both'],
                       help='Strategy to optimize')
    
    args = parser.parse_args()
    
    optimizer = StrategyOptimizer()
    
    if args.strategy in ['ema', 'both']:
        ema_results = optimizer.optimize_ema_strategy()
        print("\n✅ EMA optimization complete!")
    
    if args.strategy in ['momentum', 'both']:
        print("\n")
        momentum_results = optimizer.optimize_momentum_strategy()
        print("\n✅ Momentum optimization complete!")
