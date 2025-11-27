# Estrategia Neuronal con Aprendizaje Continuo

Sistema de trading basado en redes neuronales CNN-LSTM con capacidad de aprendizaje continuo.

## 🎯 Características

- **Arquitectura ligera CNN-LSTM** optimizada para entrenamiento en CPU
- **Aprendizaje continuo** - Se adapta automáticamente a nuevas condiciones de mercado
- **Feature engineering automático** - Calcula indicadores técnicos y features de precio
- **Sistema de versionado** - Mantiene checkpoints y permite revertir modelos
- **Backtesting integrado** - Valida rendimiento antes de trading real
- **Uso eficiente de datos** - Usa sistema de caché existente

## 📋 Requisitos

### Instalar dependencias

```bash
pip install tensorflow scikit-learn joblib
```

O si solo tienes CPU (más ligero):

```bash
pip install tensorflow-cpu scikit-learn joblib
```

## 🚀 Uso

### 1. Entrenamiento Inicial

Entrena el modelo desde cero con datos históricos cacheados:

```bash
# Entrenar con símbolos por defecto (ETH, SOL, BNB)
python neural_strategy.py --mode train

# Entrenar con símbolos específicos (recomendado)
python neural_strategy.py --mode train --symbols ETH/USDT SOL/USDT BNB/USDT

# Especificar número de épocas
python neural_strategy.py --mode train --epochs 50
```

**Tiempo estimado**: 30-60 minutos para entrenamiento inicial (CPU).

**Salida**:
- Modelo guardado en `models/neural_model_v1.keras`
- Scaler en `models/scaler_v1.pkl`
- Métricas en `models/metrics_v1.json`

### 2. Generar Predicciones

Obtén señales de trading en tiempo real:

```bash
# Predicción para un símbolo
python neural_strategy.py --mode predict --symbol ETH/USDT

# Usar versión específica del modelo
python neural_strategy.py --mode predict --symbol BTC/USDT --version 1
```

**Salida ejemplo**:
```
============================================================
Señal para ETH/USDT
============================================================
📊 SEÑAL: BUY
🎯 Confianza: 68.3%

📈 Probabilidades:
   SELL: 12.5%
   HOLD: 19.2%
   BUY: 68.3%
============================================================
```

### 3. Backtesting

Valida el rendimiento del modelo en datos históricos:

```bash
# Backtest de un símbolo
python neural_backtest.py --symbol ETH/USDT

# Múltiples símbolos
python neural_backtest.py --symbols ETH/USDT BTC/USDT SOL/USDT

# Especificar período
python neural_backtest.py --symbol ETH/USDT --start-date 2024-01-01 --end-date 2024-12-31

# Capital personalizado
python neural_backtest.py --symbol ETH/USDT --capital 100
```

**Métricas reportadas**:
- Total de operaciones
- Win Rate (tasa de aciertos)
- ROI (retorno sobre inversión)
- Max Drawdown (máxima pérdida)
- Sharpe Ratio
- Profit/Loss promedio

### 4. Test de Features

Prueba la extracción de features sin entrenar:

```bash
python neural_strategy.py --mode test --symbol ETH/USDT
```

## ⚙️ Configuración

Todos los parámetros están en `neural_config.py`:

### Arquitectura del Modelo

```python
LOOKBACK_WINDOW = 60     # Velas de contexto
CNN_FILTERS = [32, 64]   # Filtros CNN
LSTM_UNITS = 50          # Unidades LSTM
DENSE_UNITS = [32, 16]   # Capas densas
```

### Entrenamiento

```python
INITIAL_EPOCHS = 100      # Épocas entrenamiento inicial
INCREMENTAL_EPOCHS = 15   # Épocas reentrenamiento
BATCH_SIZE = 32           # Tamaño de batch
LEARNING_RATE = 0.0001    # Tasa de aprendizaje (optimizada)
```

**Nota**: Learning Rate bajo (0.0001) previene colapso del modelo durante entrenamiento con class weights.

### Aprendizaje Continuo

```python
RETRAIN_INTERVAL_HOURS = 24         # Reentrenar cada 24h
MIN_PERFORMANCE_THRESHOLD = 0.52    # Win rate mínimo 52%
MIN_SHARPE_RATIO = 0.5              # Sharpe mínimo
MAX_DRAWDOWN_THRESHOLD = 0.20       # Max drawdown 20%
```

### Señales de Trading

```python
MIN_CONFIDENCE_BUY = 0.35   # 35% confianza para BUY (optimizado)
MIN_CONFIDENCE_SELL = 0.35  # 35% confianza para SELL (optimizado)
```

**Nota**: Umbral de 0.35 filtra señales débiles manteniendo suficiente actividad de trading.

### Labeling de Datos

```python
LABEL_LOOKAHEAD = 5           # Velas hacia adelante para etiquetar
LABEL_PROFIT_THRESHOLD = 0.02  # 2% ganancia → BUY
LABEL_LOSS_THRESHOLD = -0.02   # -2% pérdida → SELL
```

## 📊 Features Extraídas

### Indicadores Técnicos
- **EMA Fast** (12)
- **EMA Slow** (26)
- **EMA Trend** (200)
- **RSI** (14)
- **ATR** (14)
- **ADX** (14)

### Features de Precio
- Returns (retornos porcentuales)
- Log Returns
- Volatilidad (rolling std)
- High-Low Ratio
- Open-Close Ratio
- Volume Change

## 🏗️ Arquitectura del Modelo

```
Input: (60 velas, ~18 features)
    ↓
CNN 1D (32 filtros) → BatchNorm → MaxPool
    ↓
CNN 1D (64 filtros) → BatchNorm → MaxPool
    ↓
LSTM (50 unidades, dropout 0.2)
    ↓
Dense (32) → Dropout (0.3)
    ↓
Dense (16) → Dropout (0.3)
    ↓
Output: Softmax(3) → [SELL, HOLD, BUY]
```

**Ventajas**:
- CNN detecta patrones locales en precios
- LSTM captura dependencias temporales
- Arquitectura compacta (~50K parámetros)
- Optimizada para CPU

## 📁 Estructura de Archivos

```
bot/
├── neural_strategy.py      # Sistema principal
├── neural_config.py        # Configuración
├── neural_backtest.py      # Backtesting
├── models/                 # Modelos entrenados
│   ├── neural_model_v1.keras
│   ├── scaler_v1.pkl
│   ├── metrics_v1.json
│   ├── checkpoints/
│   └── logs/
└── data/                   # Cache de datos (ya existente)
```

## 🔄 Aprendizaje Continuo (Futuro)

El sistema está diseñado para aprendizaje continuo, aunque esta funcionalidad está parcialmente implementada:

1. **Reentrenamiento periódico** (24h)
2. **Evaluación automática** de rendimiento
3. **Reversión a checkpoint** si degrada
4. **Actualización incremental** sin entrenar desde cero

Para habilitar:
```bash
python neural_strategy.py --mode continuous
```

⚠️ **Nota**: Modo continuo en desarrollo.

## 📈 Workflow Recomendado

### Primera vez

1. **Entrenar modelo inicial**:
   ```bash
   python neural_strategy.py --mode train
   ```

2. **Ejecutar backtest para validar**:
   ```bash
   python neural_backtest.py --symbol ETH/USDT --start-date 2024-01-01
   ```

3. **Revisar métricas esperadas**:
   - Win Rate > 45%
   - ROI > 20% (anual)
   - Sharpe Ratio > 0.5
   - Max Drawdown < 50%

4. **Si es satisfactorio, usar en predicciones**:
   ```bash
   python neural_strategy.py --mode predict --symbol ETH/USDT
   ```

### Mantenimiento

- **Reentrenar periódicamente** con nuevos datos (ej: cada semana)
- **Comparar versiones** mediante backtest
- **Actualizar configuración** según resultados

## 🎛️ Integración con Bots Existentes

La estrategia está diseñada como módulo independiente. Para integrar:

```python
from neural_strategy import NeuralStrategy

# En tu bot
strategy = NeuralStrategy()
signal = strategy.get_signal('ETH/USDT')

if signal['signal'] == 'BUY' and signal['confidence'] > 0.65:
    # Ejecutar compra
    print(f"Comprar ETH/USDT (confianza: {signal['confidence']:.2%})")
```

## ⚠️ Limitaciones y Consideraciones

1. **Entrenamiento inicial lento** (~30-60 min en CPU)
   - Solución: Usar GPU o tensorflow-cpu optimizado

2. **Necesita datos históricos suficientes**
   - Mínimo: 1000 muestras (~5000 velas)
   - Recomendado: 6-12 meses de histórico

3. **Rendimiento no garantizado**
   - Los mercados cambian constantemente
   - Siempre validar con backtest primero
   - Usar gestión de riesgo adecuada

4. **Puede aprender ruido**
   - Por eso implementamos validación estricta
   - Sistema de reversión a checkpoints
   - Métricas de rendimiento mínimo

## 🔧 Troubleshooting

### Error: TensorFlow no instalado
```bash
pip install tensorflow-cpu scikit-learn joblib
```

### Error: Datos insuficientes
```bash
# Asegúrate de tener caché actualizado
python -c "from data_cache import DataCache; cache = DataCache(); cache.get_data('ETH/USDT', force_update=True)"
```

### Model no encontrado
```bash
# Entrena primero
python neural_strategy.py --mode train
```

### Entrenamiento muy lento
- Usa `tensorflow-cpu` en vez de `tensorflow` completo
- Reduce `INITIAL_EPOCHS` en `neural_config.py`
- Reduce `LOOKBACK_WINDOW` (ej: de 60 a 40)

## 📊 Resultados de Backtest Validados

### ETH/USDT (2024-2025, Timeframe 4h)

| Métrica | Valor |
|---------|-------|
| **Total Trades** | 113 |
| **Win Rate** | 47.79% |
| **ROI** | **32.06%** |
| **Max Drawdown** | 49.54% |
| **Sharpe Ratio** | 0.55 |
| **Período** | 2024-01-01 a 2025-11-26 |

### Características de la Estrategia Optimizada

1. **Trailing Stop**: 3% (activación cuando ganancia >1%)
2. **Stop Loss Fijo**: 4% (red de seguridad)
3. **Class Weights**: Automático (balanceo de clases minoritarias)
4. **Filtro de Confianza**: 0.35 (equilibrio calidad/cantidad)

⚠️ **Nota**: Resultados pasados no garantizan rendimiento futuro. Usar gestión de riesgo.

## 🎓 Referencias

- **CNN 1D**: Detecta patrones en series temporales
- **LSTM**: Redes con memoria para secuencias
- **Transfer Learning**: Entrenar con múltiples pares
- **Online Learning**: Actualización incremental de pesos

## 📝 Próximos Pasos

- [ ] Implementar modo continuo completo
- [ ] Optimización de hiperparámetros (grid search)
- [ ] Ensemble de modelos
- [ ] Features adicionales (volumen detallado, order book)
- [ ] Dashboard de monitoreo
- [ ] Integración directa con bot_production.py

## 📄 Licencia

Parte del proyecto de trading bot. Uso bajo tu propia responsabilidad.

---

**¿Preguntas?** Revisa la configuración en `neural_config.py` o ejecuta:
```bash
python neural_strategy.py --help
```
