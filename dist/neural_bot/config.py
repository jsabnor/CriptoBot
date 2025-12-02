"""
Configuración del Sistema Neural de Trading

Define todos los parámetros del modelo, entrenamiento y aprendizaje continuo.
Configuración modular y bien documentada para fácil ajuste de hiperparámetros.
"""

from pathlib import Path


class NeuralConfig:
    """Configuración centralizada de la estrategia neuronal"""
    
    # ================== ARQUITECTURA DEL MODELO ==================
    
    # Ventana de lookback (cuántas velas históricas usar para predecir)
    LOOKBACK_WINDOW = 60  # 60 velas para contexto temporal
    
    # CNN: Detecta patrones locales en precios
    CNN_FILTERS = [64, 128]  # Filtros por capa convolucional
    CNN_KERNEL_SIZE = 3       # Tamaño del kernel
    
    # LSTM: Captura dependencias temporales
    LSTM_UNITS = 64          # Unidades LSTM
    LSTM_DROPOUT = 0.2        # Dropout para prevenir overfitting
    
    # Attention Layer
    USE_ATTENTION = False    # Desactivado - causa problemas de serialización
    ATTENTION_UNITS = 64     # Unidades en attention layer
    
    # Capas densas finales
    DENSE_UNITS = [128, 64]  # Capas densas antes de la salida
    DENSE_DROPOUT = 0.5       # Dropout en capas densas
    
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
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'stoch_k': 14,
        'stoch_d': 3,
        'cci': 20,
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
    INITIAL_EPOCHS = 150      # Épocas para entrenamiento inicial
    BATCH_SIZE = 32           # Batch size optimizado para pocas muestras
    LEARNING_RATE = 0.0005    # Learning rate inicial
    
    # Learning Rate Schedule
    USE_LR_SCHEDULE = True    # Reduce LR cuando no mejora
    LR_PATIENCE = 10          # Épocas sin mejora antes de reducir LR
    LR_FACTOR = 0.5           # Factor de reducción de LR
    LR_MIN = 0.00001          # LR mínimo
    
    # Entrenamiento incremental
    INCREMENTAL_EPOCHS = 15   # Épocas para reentrenamiento
    
    # Validación
    VALIDATION_SPLIT = 0.2    # 20% de datos para validación (más para training)
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 20  # Parar si no mejora en N épocas
    
    # ================== APRENDIZAJE CONTINUO ==================
    
    # Frecuencia de reentrenamiento
    RETRAIN_INTERVAL_HOURS = 24  # Reentrenar cada 24 horas
    
    # Evaluación de rendimiento
    PERFORMANCE_WINDOW_DAYS = 30         # Evaluar últimos 30 días
    MIN_PERFORMANCE_THRESHOLD = 0.52     # Win rate mínimo (52%)
    MIN_SHARPE_RATIO = 0.5               # Sharpe ratio mínimo
    MAX_DRAWDOWN_THRESHOLD = 0.20        # Max drawdown aceptable (20%)
    
    # Número mínimo de trades para evaluar
    MIN_TRADES_FOR_EVALUATION = 10
    
    # ================== GESTIÓN DE MODELOS ==================
    
    # Directorios
    MODELS_DIR = 'models'                    # Directorio raíz de modelos
    CHECKPOINTS_DIR = 'models/checkpoints'   # Checkpoints durante entrenamiento
    LOGS_DIR = 'models/logs'                 # Logs de TensorBoard
    
    # Nombres de archivos para sistema legacy (version-based)
    MODEL_NAME_FORMAT = 'neural_model_v{version}.keras'
    CONFIG_NAME_FORMAT = 'scaler_v{version}.pkl'
    METRICS_NAME_FORMAT = 'metrics_v{version}.json'
    
    # Nuevo sistema de gestión de modelos (named-based)
    MODELS_INDEX_FILE = 'models/models_index.json'  # Índice de modelos
    
    # Versionado legacy
    MAX_VERSIONS_TO_KEEP = 5  # Mantener últimas 5 versiones
    
    # ================== SEÑALES DE TRADING ==================
    
    # Umbral de confianza para generar señal
    MIN_CONFIDENCE_BUY = 0.30    # 30% confianza mínima para BUY (permisivo)
    MIN_CONFIDENCE_SELL = 0.35   # 35% confianza mínima para SELL (selectivo)
    
    # Clases de señal
    CLASS_LABELS = {
        0: 'SELL',
        1: 'HOLD',
        2: 'BUY'
    }
    
    # Pesos de clases para balanceo durante entrenamiento
    # Aumentar peso de BUY agresivamente para combatir HOLD bias
    CLASS_WEIGHTS = {
        0: 1.0,   # SELL - peso normal
        1: 0.65,   # HOLD - REDUCIR MÁS (combatir bias, era 0.7)
        2: 1.5,   # BUY - AUMENTAR MÁS (era 1.5, ahora 2x)
    }
    
    # ================== DATA ==================
    
    # Símbolos para entrenamiento por defecto
    DEFAULT_SYMBOLS = [
        'SOL/USDT'
    ]
    
    # Timeframe por defecto
    DEFAULT_TIMEFRAME = '4h'
    
    # Mínimo de datos para entrenar
    MIN_TRAIN_SAMPLES = 3000  # Al menos 3000 muestras (ajustado para especialistas)
    
    # ================== LABELING ==================
    
    # Estrategia para etiquetar datos históricos
    # Mira N velas hacia adelante para determinar si fue buena operación
    LABEL_LOOKAHEAD = 6              # Mirar 6 velas adelante
    LABEL_PROFIT_THRESHOLD = 0.02    # 2% ganancia → BUY label (era 1.2%)
    LABEL_LOSS_THRESHOLD = -0.02     # -2% pérdida → SELL label (era -1.8%)
    # Entre umbrales → HOLD label
    
    # ================== OPTIMIZACIÓN ==================
    
    # Optimizador
    OPTIMIZER = 'adam'
    
    # Loss function
    LOSS_FUNCTION = 'sparse_categorical_crossentropy'
    
    # Métricas
    METRICS = ['accuracy']
    
    # ================== DEBUG ==================
    
    VERBOSE = 1          # Verbosity level
    RANDOM_SEED = 42     # Seed para reproducibilidad
    
    # ================== MÉTODOS HELPER ==================
    
    @classmethod
    def ensure_directories(cls):
        """Crea los directorios necesarios si no existen"""
        Path(cls.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, name):
        """
        Obtiene la ruta completa de un modelo por nombre
        
        Args:
            name: Nombre del modelo (ej: 'eth_optimized')
        
        Returns:
            Path: Ruta al directorio del modelo
        """
        return Path(cls.MODELS_DIR) / name
    
    @classmethod
    def validate_config(cls):
        """Valida que la configuración sea coherente"""
        issues = []
        
        if cls.LOOKBACK_WINDOW < 10:
            issues.append("LOOKBACK_WINDOW muy pequeño (< 10)")
        
        if cls.VALIDATION_SPLIT < 0 or cls.VALIDATION_SPLIT >= 1:
            issues.append("VALIDATION_SPLIT debe estar entre 0 y 1")
        
        if cls.MIN_CONFIDENCE_BUY < 0 or cls.MIN_CONFIDENCE_BUY > 1:
            issues.append("MIN_CONFIDENCE_BUY debe estar entre 0 y 1")
        
        if cls.MIN_CONFIDENCE_SELL < 0 or cls.MIN_CONFIDENCE_SELL > 1:
            issues.append("MIN_CONFIDENCE_SELL debe estar entre 0 y 1")
        
        if issues:
            print("⚠️ Problemas de configuración detectados:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True


# Instancia global para fácil importación
config = NeuralConfig()

# Asegurar directorios al importar
config.ensure_directories()
