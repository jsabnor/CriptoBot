"""
Strategy Tester - Prueba diferentes estrategias de trading

Permite probar estrategias alternativas usando datos en caché:
- RSI + MACD
- Bollinger Bands + Volume
- EMA Crossover
- Ichimoku Cloud
- Custom strategies

Uso:
    python strategy_tester.py --strategy rsi_macd
    python strategy_tester.py --strategy bollinger
    python strategy_tester.py --strategy all
"""

import pandas as pd
import numpy as np
from datetime import datetime
from data_cache import DataCache
import warnings
warnings.filterwarnings('ignore')

class StrategyTester:
    """Tester para diferentes estrategias de trading"""
    
    def __init__(self, initial_capital=200, capital_per_pair=50):
        self.initial_capital = initial_capital
        self.capital_per_pair = capital_per_pair
        self.commission = 0.001  # 0.1%
        self.cache = DataCache()
        
    def calculate_rsi(self, df, period=14):
        """Calcula RSI"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self, df, period=20, std=2):
        """Calcula Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    def calculate_ema(self, df, period):
        """Calcula EMA"""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    # ==================== ESTRATEGIAS ====================
    
    def strategy_rsi_macd(self, df):
        """
        Estrategia RSI + MACD
        Compra: RSI < 30 y MACD cruza al alza
        Venta: RSI > 70 o MACD cruza a la baja
        """
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df, 14)
        df['macd'], df['signal'], df['histogram'] = self.calculate_macd(df)
        
        # Señales
        df['buy_signal'] = (
            (df['rsi'] < 35) & 
            (df['macd'] > df['signal']) & 
            (df['macd'].shift(1) <= df['signal'].shift(1))
        )
        
        df['sell_signal'] = (
            (df['rsi'] > 65) | 
            ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1)))
        )
        
        return df
    
    def strategy_bollinger_volume(self, df):
        """
        Estrategia Bollinger Bands + Volume
        Compra: Precio toca banda inferior con volumen alto
        Venta: Precio toca banda superior
        """
        df = df.copy()
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df, 20, 2)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Señales
        df['buy_signal'] = (
            (df['close'] <= df['bb_lower']) & 
            (df['volume'] > df['volume_ma'] * 1.5)
        )
        
        df['sell_signal'] = (df['close'] >= df['bb_upper'])
        
        return df
    
    def strategy_ema_crossover(self, df):
        """
        Estrategia EMA Crossover
        Compra: EMA rápida cruza EMA lenta al alza
        Venta: EMA rápida cruza EMA lenta a la baja
        """
        df = df.copy()
        df['ema_fast'] = self.calculate_ema(df, 12)
        df['ema_slow'] = self.calculate_ema(df, 26)
        
        # Señales
        df['buy_signal'] = (
            (df['ema_fast'] > df['ema_slow']) & 
            (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        )
        
        df['sell_signal'] = (
            (df['ema_fast'] < df['ema_slow']) & 
            (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        )
        
        return df
    
    def strategy_mean_reversion(self, df):
        """
        Estrategia Mean Reversion
        Compra: Precio está 2 desviaciones estándar por debajo de la media
        Venta: Precio vuelve a la media
        """
        df = df.copy()
        df['sma'] = df['close'].rolling(window=20).mean()
        df['std'] = df['close'].rolling(window=20).std()
        
        # Señales
        df['buy_signal'] = (df['close'] < df['sma'] - 2 * df['std'])
        df['sell_signal'] = (df['close'] > df['sma'])
        
        return df
    
    def strategy_momentum(self, df):
        """
        Estrategia Momentum
        Compra: ROC positivo y creciente
        Venta: ROC negativo
        """
        df = df.copy()
        period = 14
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        df['roc_ma'] = df['roc'].rolling(window=5).mean()
        
        # Señales
        df['buy_signal'] = (
            (df['roc'] > 0) & 
            (df['roc'] > df['roc_ma']) &
            (df['roc'].shift(1) <= df['roc_ma'].shift(1))
        )
        
        df['sell_signal'] = (df['roc'] < 0)
        
        return df
    
    # ==================== BACKTESTING ====================
    
    def backtest_strategy(self, symbol, strategy_func, start_date='2020-01-01'):
        """Ejecuta backtest de una estrategia"""
        # Cargar datos
        df = self.cache.get_data(symbol, '4h')
        if df is None:
            return None
        
        # Filtrar por fecha
        df = df[df['timestamp'] >= start_date].copy()
        
        # Aplicar estrategia
        df = strategy_func(df)
        
        # Simular trading
        capital = self.capital_per_pair
        position = None
        trades = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Compra
            if row['buy_signal'] and position is None:
                entry_price = row['close']
                qty = (capital * 0.95) / entry_price  # 95% del capital
                cost = qty * entry_price * (1 + self.commission)
                
                if cost <= capital:
                    position = {
                        'entry_price': entry_price,
                        'qty': qty,
                        'entry_date': row['timestamp']
                    }
                    capital -= cost
            
            # Venta
            elif row['sell_signal'] and position is not None:
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
                    'roi': roi
                })
                
                capital += proceeds
                position = None
        
        # Cerrar posición abierta si existe
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
                'roi': roi
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
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0
        
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
        
        return {
            'symbol': symbol,
            'total_trades': len(trades_df),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_roi': total_roi,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_dd,
            'final_capital': capital,
            'trades': trades_df
        }
    
    def test_all_strategies(self, symbols=['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT']):
        """Prueba todas las estrategias en todos los símbolos"""
        strategies = {
            'RSI + MACD': self.strategy_rsi_macd,
            'Bollinger + Volume': self.strategy_bollinger_volume,
            'EMA Crossover': self.strategy_ema_crossover,
            'Mean Reversion': self.strategy_mean_reversion,
            'Momentum': self.strategy_momentum
        }
        
        results = []
        
        print("=" * 80)
        print("TESTING ALL STRATEGIES")
        print("=" * 80)
        print()
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n{'='*80}")
            print(f"📊 STRATEGY: {strategy_name}")
            print(f"{'='*80}\n")
            
            strategy_results = []
            
            for symbol in symbols:
                print(f"Testing {symbol}...", end=' ')
                result = self.backtest_strategy(symbol, strategy_func)
                
                if result:
                    strategy_results.append(result)
                    print(f"✓ {result['total_trades']} trades, ROI: {result['total_roi']:.2f}%")
                else:
                    print("✗ No trades")
            
            if strategy_results:
                # Calcular promedios
                avg_roi = np.mean([r['total_roi'] for r in strategy_results])
                avg_win_rate = np.mean([r['win_rate'] for r in strategy_results])
                avg_dd = np.mean([r['max_drawdown'] for r in strategy_results])
                total_trades = sum([r['total_trades'] for r in strategy_results])
                
                results.append({
                    'strategy': strategy_name,
                    'avg_roi': avg_roi,
                    'avg_win_rate': avg_win_rate,
                    'avg_drawdown': avg_dd,
                    'total_trades': total_trades,
                    'details': strategy_results
                })
                
                print(f"\n📈 Summary:")
                print(f"   Avg ROI: {avg_roi:.2f}%")
                print(f"   Avg Win Rate: {avg_win_rate:.1f}%")
                print(f"   Avg Drawdown: {avg_dd:.2f}%")
                print(f"   Total Trades: {total_trades}")
        
        # Ranking
        print(f"\n\n{'='*80}")
        print("🏆 STRATEGY RANKING (by ROI)")
        print(f"{'='*80}\n")
        
        results_sorted = sorted(results, key=lambda x: x['avg_roi'], reverse=True)
        
        for i, result in enumerate(results_sorted, 1):
            print(f"{i}. {result['strategy']}")
            print(f"   ROI: {result['avg_roi']:.2f}% | Win Rate: {result['avg_win_rate']:.1f}% | DD: {result['avg_drawdown']:.2f}%")
            print()
        
        return results_sorted


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test different trading strategies')
    parser.add_argument('--strategy', type=str, default='all', 
                       choices=['all', 'rsi_macd', 'bollinger', 'ema', 'mean_reversion', 'momentum'],
                       help='Strategy to test')
    parser.add_argument('--symbols', type=str, nargs='+', 
                       default=['ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT'],
                       help='Symbols to test')
    
    args = parser.parse_args()
    
    tester = StrategyTester()
    
    if args.strategy == 'all':
        results = tester.test_all_strategies(args.symbols)
    else:
        strategy_map = {
            'rsi_macd': tester.strategy_rsi_macd,
            'bollinger': tester.strategy_bollinger_volume,
            'ema': tester.strategy_ema_crossover,
            'mean_reversion': tester.strategy_mean_reversion,
            'momentum': tester.strategy_momentum
        }
        
        strategy_func = strategy_map[args.strategy]
        
        for symbol in args.symbols:
            print(f"\nTesting {args.strategy} on {symbol}...")
            result = tester.backtest_strategy(symbol, strategy_func)
            
            if result:
                print(f"✓ Trades: {result['total_trades']}")
                print(f"✓ Win Rate: {result['win_rate']:.1f}%")
                print(f"✓ Total ROI: {result['total_roi']:.2f}%")
                print(f"✓ Max DD: {result['max_drawdown']:.2f}%")
            else:
                print("✗ No trades generated")
