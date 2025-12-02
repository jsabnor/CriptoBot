"""
Configuración del sistema de estrategia neuronal

Define todos los parámetros del modelo, entrenamiento y 
aprendizaje continuo.
"""

class NeuralConfig:
    """Configuración centralizada de la estrategia neuronal"""
    
    # ================== ARQUITECTURA DEL MODELO ==================
    
    # Ventana de lookback (cuántas velas históricas usar para predecir)
    LOOKBACK_WINDOW = 60  # 60 velas para contexto temporal
    
    # CNN: Detecta patrones locales en precios (OPTIMIZADO)
    CNN_FILTERS = [128, 256]  # Filtros por capa convolucional (aumentado)
    CNN_KERNEL_SIZE = 3      # Tamaño del kernel
    
    # LSTM: Captura dependencias temporales (OPTIMIZADO)
    LSTM_UNITS = 128         # Unidades LSTM (aumentado para mayor capacidad)
    LSTM_DROPOUT = 0.2      # Dropout para prevenir overfitting
    
    # Attention Layer (NUEVO)
    USE_ATTENTION = True    # Añade attention mechanism después de LSTM
    ATTENTION_UNITS = 128   # Unidades en attention layer
    
    # Capas densas finales
    DENSE_UNITS = [256, 128]  # Capas densas antes de la salida (aumentado)
    DENSE_DROPOUT = 0.3
    
    # Salida: 3 clases (BUY, SELL, HOLD)
    NUM_CLASSES = 3
    
    # ================== FEATURES ==================
    
    # Indicadores técnicos a calcular
    TECHNICAL_INDICATORS = {
        'ema_fast': 12,
        'ema_slow': 26,
        'ema_trend': 200,
        'rsi': 14,
        'atr': 14,
        'adx': 14,
        'macd_fast': 12,      # MACD fast period
        'macd_slow': 26,      # MACD slow period
        'macd_signal': 9,     # MACD signal period
        'bb_period': 20,      # Bollinger Bands period
        'bb_std': 2,          # Bollinger Bands standard deviation
        'stoch_k': 14,        # Stochastic %K period
        'stoch_d': 3,         # Stochastic %D period
        'cci': 20,            # CCI period
    }
    
    # Features de precio (se calculan automáticamente)
    PRICE_FEATURES = [
        'returns',           # Retornos porcentuales
        'log_returns',       # Log returns
        'volatility',        # Volatilidad rolling
        'hl_ratio',          # High-Low ratio
        'oc_ratio',          # Open-Close ratio
        'volume_change',     # Cambio en volumen
    ]
    
    # Features de volumen avanzados
    VOLUME_FEATURES = [
        'vwap',              # Volume Weighted Average Price
        'obv',               # On Balance Volume
        'volume_ratio',      # Volume vs average
    ]
    
    # Features cruzados (derivados de otros indicadores)
    CROSS_FEATURES = [
        'ema_cross',         # EMA fast vs slow (crossover signal)
        'price_to_ema_fast', # Distance from price to EMA fast
        'price_to_ema_slow', # Distance from price to EMA slow
    ]
    
    # ================== ENTRENAMIENTO ==================
    
    # Entrenamiento inicial
    INITIAL_EPOCHS = 150         # Épocas para entrenamiento inicial (aumentado)
    BATCH_SIZE = 64              # Batch size optimizado
    LEARNING_RATE = 0.0005        # Learning rate inicial (con schedule)
    
    # Learning Rate Schedule (NUEVO)
    USE_LR_SCHEDULE = True       # Reduce LR cuando no mejora
    LR_PATIENCE = 10             # Épocas sin mejora antes de reducir LR
    LR_FACTOR = 0.5              # Factor de reducción de LR
    LR_MIN = 0.00001             # LR mínimo
    
    # Entrenamiento incremental
    INCREMENTAL_EPOCHS = 15    # Épocas para reentrenamiento
    
    # Validación
    VALIDATION_SPLIT = 0.3      # 30% de datos para validación
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20  # Parar si no mejora en N épocas (ajustado)
    
    # ================== APRENDIZAJE CONTINUO ==================
    
    # Frecuencia de reentrenamiento
    RETRAIN_INTERVAL_HOURS = 24  # Reentrenar cada 24 horas
    
    # Evaluación de rendimiento
    PERFORMANCE_WINDOW_DAYS = 30        # Evaluar últimos 30 días
    MIN_PERFORMANCE_THRESHOLD = 0.52    # Win rate mínimo (52%)
    MIN_SHARPE_RATIO = 0.5              # Sharpe ratio mínimo
    MAX_DRAWDOWN_THRESHOLD = 0.20       # Max drawdown aceptable (20%)
    
    # Número mínimo de trades para evaluar
    MIN_TRADES_FOR_EVALUATION = 10
    
    # ================== GESTIÓN DE MODELOS ==================
    
    # Directorios
    MODELS_DIR = 'models'               # Directorio de modelos
    CHECKPOINTS_DIR = 'models/checkpoints'
    LOGS_DIR = 'models/logs'
    
    # Formato de archivos
    MODEL_NAME_FORMAT = 'neural_model_v{version}.keras'
    CONFIG_NAME_FORMAT = 'scaler_v{version}.pkl'
    METRICS_NAME_FORMAT = 'metrics_v{version}.json'
    
    # Versionado
    MAX_VERSIONS_TO_KEEP = 5    # Mantener últimas 5 versiones
    
    # ================== SEÑALES DE TRADING ==================
    
    # Umbral de confianza para generar señal
    MIN_CONFIDENCE_BUY = 0.32    # 35% confianza (más selectivo)
    MIN_CONFIDENCE_SELL = 0.32   # 35% confianza (más selectivo)
    
    # Clases de señal
    CLASS_LABELS = {
        0: 'SELL',
        1: 'HOLD',
        2: 'BUY'
    }
    
    # ================== DATA ==================
    
    # Símbolos para entrenamiento
    DEFAULT_SYMBOLS = [
        'ETH/USDT',
        'SOL/USDT',
        'XRP/USDT',
        'ADA/USDT',
        'BNB/USDT'
    ]
    
    # Timeframe
    DEFAULT_TIMEFRAME = '1h'
    
    # Mínimo de datos para entrenar
    MIN_TRAIN_SAMPLES = 5000    # Al menos 1000 muestras
    
    # ================== LABELING ==================
    
    # Estrategia para etiquetar datos históricos
    # Mira N velas hacia adelante para determinar si fue buena operación
    LABEL_LOOKAHEAD = 6         # Mirar 5 velas adelante
    LABEL_PROFIT_THRESHOLD = 0.012  # 2% ganancia → BUY label
    LABEL_LOSS_THRESHOLD = -0.018   # -2% pérdida → SELL label
    # Entre -2% y +2% → HOLD label
    
    # ================== OPTIMIZACIÓN ==================
    
    # Optimizador
    OPTIMIZER = 'adam'
    
    # Loss function
    LOSS_FUNCTION = 'sparse_categorical_crossentropy'
    
    # Métricas
    METRICS = ['accuracy']
    
    # ================== DEBUG ==================
    
    VERBOSE = 1                 # Verbosity level
    RANDOM_SEED = 42            # Seed para reproducibilidad


# Instancia global para fácil importación
config = NeuralConfig()
