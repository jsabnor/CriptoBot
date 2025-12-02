"""
Debug script para ver qué está prediciendo el modelo
"""
import numpy as np
from neural_bot import NeuralStrategy
from data_cache import DataCache

# Cargar estrategia
strategy = NeuralStrategy()
cache = DataCache()

# Cargar datos
df = cache.get_data('ETH/USDT', '4h')
df_recent = df.tail(100)  # últimas 100 velas

print("\n" + "="*60)
print("🔍 ANÁLISIS DE PREDICCIONES DEL MODELO")
print("="*60)

# Extraer features
X = strategy.feature_extractor.extract_features(df_recent, fit_scaler=False)
X_seq = strategy.feature_extractor.create_sequences(X)

print(f"\n📊 Datos: {len(X_seq)} secuencias")

# Predecir
predictions = strategy.model.predict(X_seq)

# Analizar predicciones
print("\n📈 Distribución de predicciones:")
for i in range(3):
    count = np.sum(np.argmax(predictions, axis=1) == i)
    pct = count / len(predictions) * 100
    from neural_config import config
    print(f"   {config.CLASS_LABELS[i]}: {count} ({pct:.1f}%)")

# Ver confianzas promedio
print("\n🎯 Confianzas promedio por clase:")
for i in range(3):
    avg_conf = np.mean(predictions[:, i])
    max_conf = np.max(predictions[:, i])
    from neural_config import config
    print(f"   {config.CLASS_LABELS[i]}: avg={avg_conf:.3f}, max={max_conf:.3f}")

# Ver las 10 predicciones más confiantes para BUY
print("\n💰 Top 10 predicciones BUY (más confiantes):")
buy_probs = predictions[:, 2]  # BUY es clase 2
top_buy_indices = np.argsort(buy_probs)[-10:][::-1]

for idx in top_buy_indices:
    probs = predictions[idx]
    predicted_class = np.argmax(probs)
    from neural_config import config
    signal = config.CLASS_LABELS[predicted_class]
    print(f"   Vela {idx}: SELL={probs[0]:.3f}, HOLD={probs[1]:.3f}, BUY={probs[2]:.3f} → {signal}")

# Ver las 10 predicciones más confiantes para SELL
print("\n📉 Top 10 predicciones SELL (más confiantes):")
sell_probs = predictions[:, 0]  # SELL es clase 0
top_sell_indices = np.argsort(sell_probs)[-10:][::-1]

for idx in top_sell_indices:
    probs = predictions[idx]
    predicted_class = np.argmax(probs)
    from neural_config import config
    signal = config.CLASS_LABELS[predicted_class]
    print(f"   Vela {idx}: SELL={probs[0]:.3f}, HOLD={probs[1]:.3f}, BUY={probs[2]:.3f} → {signal}")

print("\n" + "="*60)
print("💡 RECOMENDACIÓN:")
print("   Si max BUY confidence < 0.4 → Reducir MIN_CONFIDENCE_BUY")
print("   Si max SELL confidence < 0.4 → Reducir MIN_CONFIDENCE_SELL")
print("="*60 + "\n")
