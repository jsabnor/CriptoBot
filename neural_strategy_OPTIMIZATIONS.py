"""
INSTRUCCIONES PARA APLICAR LA OPTIMIZACIÓN
============================================

Este archivo contiene las secciones modificadas de neural_strategy.py con todas las optimizaciones.

MÉTODO 1 - Reemplazo Manual (Recomendado):
1. Abre neural_strategy.py
2. Busca cada sección marcada con "# BUSCAR:" y reemplázala con el código de "# REEMPLAZAR CON:"

MÉTODO 2 - Backup y Reemplazo Completo:
1. Haz backup: cp neural_strategy.py neural_strategy_backup.py
2. Copia el contenido completo optimizado que te proporcionaré en el siguiente archivo

Las modificaciones son:
- ✅ Indicadores: MACD, Bollinger Bands, Stochastic, CCI, VWAP, OBV
- ✅ Cross Features: EMA crossover, price-to-EMA distances  
- ✅ Arquitectura: Attention Layer, BatchNorm, CNN/LSTM aumentados
- ✅ Entrenamiento: Learning Rate Schedule

============================================
"""

# ============================================
# SECCIÓN 1: calculate_technical_indicators
# ============================================

"""
# BUSCAR ESTA FUNCIÓN (líneas ~53-102):
def calculate_technical_indicators(self, df):

# REEMPLAZAR CON:
"""

def calculate_technical_indicators(self, df):
    """Calcula indicadores técnicos configurados (OPTIMIZADO)"""
    result = df.copy()
    
    # EMAs
    for name, period in config.TECHNICAL_INDICATORS.items():
        if 'ema' in name and name in ['ema_fast', 'ema_slow', 'ema_trend']:
            result[name] = result['close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    if 'rsi' in config.TECHNICAL_INDICATORS:
        period = config.TECHNICAL_INDICATORS['rsi']
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    if 'atr' in config.TECHNICAL_INDICATORS:
        period = config.TECHNICAL_INDICATORS['atr']
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result['atr'] = true_range.rolling(window=period).mean()
    
    # ADX
    if 'adx' in config.TECHNICAL_INDICATORS:
        period = config.TECHNICAL_INDICATORS['adx']
        
        high_diff = result['high'].diff()
        low_diff = -result['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        if 'atr' not in result.columns:
            result['atr'] = true_range.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / result['atr'])
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / result['atr'])
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        result['adx'] = dx.rolling(window=period).mean()
    
    # MACD (NUEVO)
    if all(k in config.TECHNICAL_INDICATORS for k in ['macd_fast', 'macd_slow', 'macd_signal']):
        fast = config.TECHNICAL_INDICATORS['macd_fast']
        slow = config.TECHNICAL_INDICATORS['macd_slow']
        signal = config.TECHNICAL_INDICATORS['macd_signal']
        
        ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
        
        result['macd_line'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd_line'].ewm(span=signal, adjust=False).mean()
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
    
    # Bollinger Bands (NUEVO)
    if 'bb_period' in config.TECHNICAL_INDICATORS and 'bb_std' in config.TECHNICAL_INDICATORS:
        period = config.TECHNICAL_INDICATORS['bb_period']
        std_dev = config.TECHNICAL_INDICATORS['bb_std']
        
        result['bb_middle'] = result['close'].rolling(window=period).mean()
        bb_std = result['close'].rolling(window=period).std()
        
        result['bb_upper'] = result['bb_middle'] + (bb_std * std_dev)
        result['bb_lower'] = result['bb_middle'] - (bb_std * std_dev)
        
        # %B: Posición relativa dentro de las bandas
        result['bb_percent'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # Bandwidth: Ancho de las bandas (volatilidad)
        result['bb_bandwidth'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    
    # Stochastic Oscillator (NUEVO)
    if 'stoch_k' in config.TECHNICAL_INDICATORS and 'stoch_d' in config.TECHNICAL_INDICATORS:
        k_period = config.TECHNICAL_INDICATORS['stoch_k']
        d_period = config.TECHNICAL_INDICATORS['stoch_d']
        
        low_min = result['low'].rolling(window=k_period).min()
        high_max = result['high'].rolling(window=k_period).max()
        
        result['stoch_k'] = 100 * (result['close'] - low_min) / (high_max - low_min)
        result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
    
    # CCI - Commodity Channel Index (NUEVO)
    if 'cci' in config.TECHNICAL_INDICATORS:
        period = config.TECHNICAL_INDICATORS['cci']
        
        tp = (result['high'] + result['low'] + result['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        result['cci'] = (tp - sma_tp) / (0.015 * mad)
    
    # VWAP - Volume Weighted Average Price (NUEVO)
    if hasattr(config, 'VOLUME_FEATURES') and 'vwap' in config.VOLUME_FEATURES:
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        result['vwap'] = (typical_price * result['volume']).cumsum() / result['volume'].cumsum()
    
    # OBV - On Balance Volume (NUEVO)
    if hasattr(config, 'VOLUME_FEATURES') and 'obv' in config.VOLUME_FEATURES:
        obv = [0]
        for i in range(1, len(result)):
            if result['close'].iloc[i] > result['close'].iloc[i-1]:
                obv.append(obv[-1] + result['volume'].iloc[i])
            elif result['close'].iloc[i] < result['close'].iloc[i-1]:
                obv.append(obv[-1] - result['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        result['obv'] = obv
    
    # Volume Ratio (NUEVO)
    if hasattr(config, 'VOLUME_FEATURES') and 'volume_ratio' in config.VOLUME_FEATURES:
        result['volume_ratio'] = result['volume'] / result['volume'].rolling(window=20).mean()
    
    return result


# ============================================
# SECCIÓN 2: calculate_price_features
# ============================================

"""
# BUSCAR ESTA FUNCIÓN (líneas ~104-132):
def calculate_price_features(self, df):

# REEMPLAZAR CON:
"""

def calculate_price_features(self, df):
    """Calcula features basadas en precio (OPTIMIZADO)"""
    result = df.copy()
    
    # Returns
    if 'returns' in config.PRICE_FEATURES:
        result['returns'] = result['close'].pct_change()
    
    # Log returns
    if 'log_returns' in config.PRICE_FEATURES:
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
    
    # Volatilidad
    if 'volatility' in config.PRICE_FEATURES:
        result['volatility'] = result['close'].pct_change().rolling(window=20).std()
    
    # High-Low ratio
    if 'hl_ratio' in config.PRICE_FEATURES:
        result['hl_ratio'] = (result['high'] - result['low']) / result['close']
    
    # Open-Close ratio
    if 'oc_ratio' in config.PRICE_FEATURES:
        result['oc_ratio'] = (result['close'] - result['open']) / result['open']
    
    # Volume change
    if 'volume_change' in config.PRICE_FEATURES:
        result['volume_change'] = result['volume'].pct_change()
    
    # CROSS FEATURES (NUEVO)
    if hasattr(config, 'CROSS_FEATURES'):
        # EMA Crossover Signal
        if 'ema_cross' in config.CROSS_FEATURES and 'ema_fast' in result.columns and 'ema_slow' in result.columns:
            result['ema_cross'] = (result['ema_fast'] - result['ema_slow']) / result['close']
        
        # Distance from price to EMA fast
        if 'price_to_ema_fast' in config.CROSS_FEATURES and 'ema_fast' in result.columns:
            result['price_to_ema_fast'] = (result['close'] - result['ema_fast']) / result['close']
        
        # Distance from price to EMA slow
        if 'price_to_ema_slow' in config.CROSS_FEATURES and 'ema_slow' in result.columns:
            result['price_to_ema_slow'] = (result['close'] - result['ema_slow']) / result['close']
    
    return result


# ============================================
# SECCIÓN 3: extract_features - Selección de features
# ============================================

"""
# BUSCAR ESTA SECCIÓN EN extract_features (líneas ~236-253):
# Seleccionar columnas de features
feature_cols = []

# OHLCV básicos
feature_cols.extend(['open', 'high', 'low', 'close', 'volume'])

# Indicadores técnicos
for name in config.TECHNICAL_INDICATORS.keys():
    if name in df_features.columns:
        feature_cols.append(name)

# Features de precio
for name in config.PRICE_FEATURES:
    if name in df_features.columns:
        feature_cols.append(name)

# Guardar nombres de features
self.feature_names = feature_cols

# REEMPLAZAR CON:
"""

# Seleccionar columnas de features
feature_cols = []

# OHLCV básicos
feature_cols.extend(['open', 'high', 'low', 'close', 'volume'])

# Indicadores técnicos base
for name in ['ema_fast', 'ema_slow', 'ema_trend', 'rsi', 'atr', 'adx']:
    if name in df_features.columns:
        feature_cols.append(name)

# MACD features
for macd_feat in ['macd_line', 'macd_signal', 'macd_histogram']:
    if macd_feat in df_features.columns:
        feature_cols.append(macd_feat)

# Bollinger Bands features  
for bb_feat in ['bb_percent', 'bb_bandwidth']:
    if bb_feat in df_features.columns:
        feature_cols.append(bb_feat)

# Stochastic features
for stoch_feat in ['stoch_k', 'stoch_d']:
    if stoch_feat in df_features.columns:
        feature_cols.append(stoch_feat)

# CCI
if 'cci' in df_features.columns:
    feature_cols.append('cci')

# Features de precio
for name in config.PRICE_FEATURES:
    if name in df_features.columns:
        feature_cols.append(name)

# Volume features (NUEVO)
if hasattr(config, 'VOLUME_FEATURES'):
    for name in config.VOLUME_FEATURES:
        if name in df_features.columns:
            feature_cols.append(name)

# Cross features (NUEVO)
if hasattr(config, 'CROSS_FEATURES'):
    for name in config.CROSS_FEATURES:
        if name in df_features.columns:
            feature_cols.append(name)

# Guardar nombres de features
self.feature_names = feature_cols


# ============================================
# SECCIÓN 4: build_model con Attention Layer
# ============================================

"""
# BUSCAR ESTA FUNCIÓN (líneas ~300-344):
def build_model(self):

# REEMPLAZAR CON:
"""

def build_model(self):
    """Construye arquitectura CNN-LSTM híbrida con Attention (OPTIMIZADO)"""
    
    # Input
    inputs = layers.Input(shape=self.input_shape)
    
    # CNN 1D para detectar patrones locales
    x = inputs
    for filters in config.CNN_FILTERS:
        x = layers.Conv1D(
            filters=filters,
            kernel_size=config.CNN_KERNEL_SIZE,
            padding='same',
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM para dependencias temporales
    if config.USE_ATTENTION:
        # Con Attention: necesita return_sequences=True
        lstm_out = layers.LSTM(
            units=config.LSTM_UNITS,
            dropout=config.LSTM_DROPOUT,
            return_sequences=True
        )(x)
        
        # BatchNormalization después de LSTM
        lstm_out = layers.BatchNormalization()(lstm_out)
        
        # Attention Mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(config.LSTM_UNITS)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention weights
        x = layers.Multiply()([lstm_out, attention])
        x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
    else:
        # Sin Attention: LSTM normal
        x = layers.LSTM(
            units=config.LSTM_UNITS,
            dropout=config.LSTM_DROPOUT,
            return_sequences=False
        )(x)
        x = layers.BatchNormalization()(x)
    
    # Capas densas
    for units in config.DENSE_UNITS:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(config.DENSE_DROPOUT)(x)
    
    # Output: 3 clases (SELL, HOLD, BUY)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    # Compilar modelo
    self.model = models.Model(inputs=inputs, outputs=outputs)
    
    self.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=config.LOSS_FUNCTION,
        metrics=config.METRICS
    )
    
    print("✅ Modelo construido (OPTIMIZADO):")
    print(f"   Input shape: {self.input_shape}")
    print(f"   CNN filters: {config.CNN_FILTERS}")
    print(f"   LSTM units: {config.LSTM_UNITS}")
    print(f"   Attention: {'Enabled' if config.USE_ATTENTION else 'Disabled'}")
    print(f"   Dense layers: {config.DENSE_UNITS}")
    print(f"   Parámetros: {self.model.count_params():,}")


# ============================================
# SECCIÓN 5: train con Learning Rate Schedule
# ============================================

"""
# BUSCAR EN LA FUNCIÓN train (líneas ~384-392):
# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss' if X_val is not None else 'loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
]

# REEMPLAZAR CON:
"""

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss' if X_val is not None else 'loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
]

# Learning Rate Schedule (NUEVO)
if config.USE_LR_SCHEDULE:
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss' if X_val is not None else 'loss',
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.LR_MIN,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    print(f"\n📉 Learning Rate Schedule activado:")
    print(f"   Initial LR: {config.LEARNING_RATE}")
    print(f"   Factor: {config.LR_FACTOR}")
    print(f"   Patience: {config.LR_PATIENCE}")
    print(f"   Min LR: {config.LR_MIN}")


"""
============================================
FIN DE LAS MODIFICACIONES
============================================

DESPUÉS DE APLICAR LOS CAMBIOS:

1. Verificar que no hay errores de sintaxis:
   python -m py_compile neural_strategy.py

2. Hacer un test rápido:
   python neural_strategy.py --mode test --symbol ETH/USDT

3. Si todo funciona, entrenar:
   python neural_strategy.py --mode train --symbols ETH/USDT SOL/USDT

============================================
"""
