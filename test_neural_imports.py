"""Test de imports del paquete neural_bot sin TensorFlow"""

import sys
print("Testing neural_bot package imports...\n")

# Test 1: config
try:
    from neural_bot.config import config, NeuralConfig
    print("✓ config import OK")
    print(f"  - LOOKBACK_WINDOW: {config.LOOKBACK_WINDOW}")
    print(f"  - DEFAULT_SYMBOLS: {len(config.DEFAULT_SYMBOLS)} symbols")
except Exception as e:
    print(f"✗ config import FAIL: {e}")

# Test 2: model_manager
try:
    from neural_bot.model_manager import ModelManager
    print("✓ ModelManager import OK")
    manager = ModelManager()
    print(f"  - Models dir: {manager.models_dir}")
except Exception as e:
    print(f"✗ ModelManager import FAIL: {e}")

try:
    import neural_bot
    print("✓ neural_bot package OK")
    print(f"  - __version__: {neural_bot.__version__}")
    print(f"  - Available exports: {', '.join(neural_bot.__all__)}")
except Exception as e:
    print(f"✗ neural_bot package FAIL: {e}")

print("\n" + "="*60)
print("✓ Tests básicos completados")
print("="*60)
print("\nNOTA: NeuralStrategy y NeuralBacktest requieren TensorFlow")
print("      para importarse completamente.")
