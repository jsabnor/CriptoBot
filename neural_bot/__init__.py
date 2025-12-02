"""
Neural Bot - Sistema de Trading con Redes Neuronales

Un sistema completo de trading basado en aprendizaje profundo que utiliza
arquitectura CNN-LSTM para predecir movimientos de mercado.

Módulos principales:
    - config: Configuración del sistema
    - model_manager: Gestión de modelos entrenados
    - strategy: Estrategia de trading neuronal
    - backtest: Sistema de backtesting
    - cli: Interfaz de línea de comandos

Uso básico:
    from neural_bot import NeuralStrategy, NeuralConfig
    
    strategy = NeuralStrategy(model_name='eth_optimized')
    signal = strategy.get_signal('ETH/USDT', '1h')
"""

__version__ = '2.0.0'
__author__ = 'CryptoBot Neural Team'

# Imports principales para facilitar el uso
from .config import NeuralConfig, config
from .model_manager import ModelManager

# TensorFlow-dependent imports (lazy loading)
try:
    from .strategy import NeuralStrategy
    from .backtest import NeuralBacktest
    __all__ = [
        'NeuralConfig',
        'config',
        'NeuralStrategy',
        'NeuralBacktest',
        'ModelManager',
    ]
except ImportError as e:
    # TensorFlow not installed, only expose non-TF components
    __all__ = [
        'NeuralConfig',
        'config',
        'ModelManager',
    ]
    print(f"⚠️ Warning: NeuralStrategy and NeuralBacktest not available (TensorFlow required)")
    print(f"   Install with: pip install tensorflow")
