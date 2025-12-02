"""
Neural Trading Strategy - Sistema de trading con aprendizaje continuo

Implementa un modelo CNN-LSTM para detección de patrones en gráficos
con capacidad de aprendizaje continuo.

Uso:
    # Entrenamiento inicial
    python neural_strategy.py --mode train --symbols ETH/USDT BTC/USDT
    
    # Predicción
    python neural_strategy.py --mode predict --symbol ETH/USDT
    
    # Aprendizaje continuo
    python neural_strategy.py --mode continuous
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    print("⚠️ TensorFlow no está instalado. Instala con: pip install tensorflow")
    print("   Para solo CPU: pip install tensorflow-cpu")
    exit(1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
from data_cache import DataCache
from neural_config import config


class FeatureExtractor:
    """Extrae y normaliza features de datos OHLCV"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def calculate_technical_indicators(self, df):
        """Calcula indicadores técnicos configurados"""
        result = df.copy()
        
        # EMAs
        for name, period in config.TECHNICAL_INDICATORS.items():
            if 'ema' in name:
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
            
            # +DM y -DM
            high_diff = result['high'].diff()
            low_diff = -result['low'].diff()
            
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            # ATR ya calculado arriba
            if 'atr' not in result.columns:
                result['atr'] = true_range.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / result['atr'])
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / result['atr'])
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            result['adx'] = dx.rolling(window=period).mean()
        
        return result
    
    def calculate_price_features(self, df):
        """Calcula features basadas en precio"""
        result = df.copy()
        
        # Returns
        if 'returns' in config.PRICE_FEATURES:
            result['returns'] = result['close'].pct_change()
        
        # Log returns
        if 'log_returns' in config.PRICE_FEATURES:
            result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Volatilidad (rolling std de returns)
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
        
        return result
    
    def extract_features(self, df, fit_scaler=False):
        """
        Extrae todas las features de un DataFrame OHLCV
        
        Args:
            df: DataFrame con columnas [timestamp, open, high, low, close, volume]
            fit_scaler: Si True, ajusta el scaler (solo para entrenamiento)
        
        Returns:
            np.array con features normalizadas, shape (n_samples, n_features)
        """
        # Calcular indicadores
        df_features = self.calculate_technical_indicators(df)
        df_features = self.calculate_price_features(df_features)
        
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
        
        # Extraer features
        df_temp = df_features[feature_cols].copy()
        
        # CRÍTICO: Manejar valores infinitos y NaN ANTES de normalizar
        # 1. Reemplazar infinitos con NaN
        df_temp = df_temp.replace([np.inf, -np.inf], np.nan)
        
        # 2. Rellenar NaN con forward fill, backward fill, y finalmente 0
        df_temp = df_temp.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 3. Convertir a numpy
        X = df_temp.values
        
        # 4. Verificar que no queden NaN o infinitos
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"⚠️ Advertencia: Aún hay NaN o infinitos después de limpieza")
            # Último recurso: reemplazar con 0
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizar
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def create_sequences(self, X, y=None):
        """
        Crea secuencias de ventanas temporales para LSTM
        
        Args:
            X: Features normalizadas (n_samples, n_features)
            y: Labels opcionales (n_samples,)
        
        Returns:
            X_seq: (n_sequences, lookback, n_features)
            y_seq: (n_sequences,) si y fue proporcionado
        """
        lookback = config.LOOKBACK_WINDOW
        
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def save_scaler(self, version):
        """Guarda el scaler entrenado"""
        path = Path(config.MODELS_DIR) / config.CONFIG_NAME_FORMAT.format(version=version)
        joblib.dump(self.scaler, path)
        print(f"💾 Scaler guardado: {path}")
    
    def load_scaler(self, version):
        """Carga un scaler guardado"""
        path = Path(config.MODELS_DIR) / config.CONFIG_NAME_FORMAT.format(version=version)
        if path.exists():
            self.scaler = joblib.load(path)
            print(f"📂 Scaler cargado: {path}")
            return True
        return False


class DataLabeler:
    """Etiqueta datos históricos para entrenamiento supervisado"""
    
    @staticmethod
    def label_data(df):
        """
        Etiqueta datos basándose en rendimiento futuro
        
        Mira N velas hacia adelante y determina si hubiera sido
        una buena operación de compra, venta o hold.
        
        Returns:
            np.array de labels: 0=SELL, 1=HOLD, 2=BUY
        """
        lookahead = config.LABEL_LOOKAHEAD
        profit_threshold = config.LABEL_PROFIT_THRESHOLD
        loss_threshold = config.LABEL_LOSS_THRESHOLD
        
        labels = []
        
        for i in range(len(df)):
            # Últimas velas no tienen suficiente futuro
            if i >= len(df) - lookahead:
                labels.append(1)  # HOLD por defecto
                continue
            
            current_price = df.iloc[i]['close']
            future_prices = df.iloc[i+1:i+lookahead+1]['close']
            
            # Calcular max ganancia y pérdida en el período futuro
            max_gain = (future_prices.max() - current_price) / current_price
            max_loss = (future_prices.min() - current_price) / current_price
            
            # Etiquetar
            if max_gain >= profit_threshold and abs(max_loss) < abs(profit_threshold) * 0.5:
                # Buena oportunidad de compra
                labels.append(2)  # BUY
            elif max_loss <= loss_threshold and max_gain < abs(loss_threshold) * 0.5:
                # Probable caída, mejor vender/evitar
                labels.append(0)  # SELL
            else:
                # Zona neutral
                labels.append(1)  # HOLD
        
        return np.array(labels)


class NeuralTradingModel:
    """Modelo CNN-LSTM para predicción de señales de trading"""
    
    def __init__(self, input_shape):
        """
        Args:
            input_shape: (lookback_window, n_features)
        """
        self.input_shape = input_shape
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Construye arquitectura CNN-LSTM híbrida"""
        
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
        x = layers.LSTM(
            units=config.LSTM_UNITS,
            dropout=config.LSTM_DROPOUT,
            return_sequences=False
        )(x)
        
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
        
        print("✅ Modelo construido:")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Parámetros: {self.model.count_params():,}")
    
    def get_summary(self):
        """Muestra resumen del modelo"""
        return self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None):
        """
        Entrena el modelo
        
        Args:
            X_train: Features de entrenamiento (n_samples, lookback, n_features)
            y_train: Labels de entrenamiento (n_samples,)
            X_val: Features de validación (opcional)
            y_val: Labels de validación (opcional)
            epochs: Número de épocas (usa config si no se especifica)
        
        Returns:
            history: Historial de entrenamiento
        """
        if epochs is None:
            epochs = config.INITIAL_EPOCHS
        
        # NUEVO: Calcular class weights para balancear clases
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        # Convertir a diccionario
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights_array)}
        
        print(f"\n⚖️ Class Weights (para balancear):")
        for cls, weight in class_weight_dict.items():
            print(f"   {config.CLASS_LABELS[cls]}: {weight:.2f}x")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Validación
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Entrenar
        print(f"\n🎓 Entrenando modelo...")
        print(f"   Samples: {len(X_train)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,  # CRÍTICO: Aplicar pesos
            verbose=config.VERBOSE
        )
        
        return history
    
    def predict(self, X):
        """
        Predice señales
        
        Args:
            X: Features (n_samples, lookback, n_features)
        
        Returns:
            predictions: Array de probabilidades (n_samples, 3)
        """
        return self.model.predict(X, verbose=0)
    
    def predict_signal(self, X):
        """
        Predice señal con etiqueta
        
        Args:
            X: Features (1, lookback, n_features) o (lookback, n_features)
        
        Returns:
            dict: {'signal': 'BUY'/'SELL'/'HOLD', 'confidence': float, 'probabilities': dict}
        """
        # Asegurar shape correcto
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        
        # Predicción
        probs = self.predict(X)[0]
        
        # Clase con mayor probabilidad
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        # Aplicar umbrales de confianza
        signal = config.CLASS_LABELS[predicted_class]
        
        if signal == 'BUY' and confidence < config.MIN_CONFIDENCE_BUY:
            signal = 'HOLD'
        elif signal == 'SELL' and confidence < config.MIN_CONFIDENCE_SELL:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'probabilities': {
                'SELL': float(probs[0]),
                'HOLD': float(probs[1]),
                'BUY': float(probs[2])
            }
        }
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo en datos de test"""
        print("\n📊 Evaluando modelo...")
        
        # Predicciones
        y_pred_probs = self.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Métricas
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['SELL', 'HOLD', 'BUY']
        ))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"\n✅ Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def save(self, version):
        """Guarda el modelo"""
        path = Path(config.MODELS_DIR) / config.MODEL_NAME_FORMAT.format(version=version)
        self.model.save(path)
        print(f"💾 Modelo guardado: {path}")
    
    def load(self, version):
        """Carga un modelo guardado"""
        path = Path(config.MODELS_DIR) / config.MODEL_NAME_FORMAT.format(version=version)
        if path.exists():
            self.model = keras.models.load_model(path)
            print(f"📂 Modelo cargado: {path}")
            return True
        return False


class ContinuousLearner:
    """Gestiona el ciclo de aprendizaje continuo"""
    
    def __init__(self):
        self.cache = DataCache()
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.current_version = 0
        self.metrics_history = []
    
    def get_latest_version(self):
        """Encuentra la versión más reciente del modelo"""
        models_dir = Path(config.MODELS_DIR)
        if not models_dir.exists():
            return 0
        
        model_files = list(models_dir.glob('neural_model_v*.keras'))
        if not model_files:
            return 0
        
        versions = []
        for f in model_files:
            try:
                version = int(f.stem.split('_v')[1])
                versions.append(version)
            except:
                continue
        
        return max(versions) if versions else 0
    
    def load_latest_model(self):
        """Carga el modelo más reciente"""
        version = self.get_latest_version()
        if version == 0:
            print("⚠️ No hay modelos guardados")
            return False
        
        self.current_version = version
        
        # Cargar scaler
        if not self.feature_extractor.load_scaler(version):
            print(f"❌ No se pudo cargar scaler v{version}")
            return False
        
        # Cargar modelo
        # Necesitamos crear modelo primero (necesita input shape)
        # Esto es un chicken-egg problem, lo resolveremos cargando datos dummy
        # En producción, guardaremos también el input_shape
        
        print(f"📂 Cargando modelo v{version}...")
        return True
    
    def prepare_training_data(self, symbols=None, timeframe='4h'):
        """
        Prepara datos de entrenamiento desde caché
        
        Args:
            symbols: Lista de símbolos (usa default si None)
            timeframe: Timeframe
        
        Returns:
            X_train, y_train, X_val, y_val
        """
        if symbols is None:
            symbols = config.DEFAULT_SYMBOLS
        
        print(f"\n📊 Preparando datos de entrenamiento...")
        print(f"   Símbolos: {symbols}")
        print(f"   Timeframe: {timeframe}")
        
        all_X = []
        all_y = []
        
        for symbol in symbols:
            print(f"\n  Procesando {symbol}...")
            
            # Cargar datos del caché
            df = self.cache.get_data(symbol, timeframe)
            
            if df is None or len(df) < config.MIN_TRAIN_SAMPLES:
                print(f"    ⚠️ Datos insuficientes ({len(df) if df is not None else 0} velas)")
                continue
            
            print(f"    ✅ {len(df)} velas cargadas")
            
            # Etiquetar datos
            y = DataLabeler.label_data(df)
            
            # Extraer features (fit solo en el primer símbolo)
            fit_scaler = len(all_X) == 0
            X = self.feature_extractor.extract_features(df, fit_scaler=fit_scaler)
            
            # Crear secuencias
            X_seq, y_seq = self.feature_extractor.create_sequences(X, y)
            
            print(f"    ✅ {len(X_seq)} secuencias creadas")
            
            all_X.append(X_seq)
            all_y.append(y_seq)
        
        # Concatenar todos los datos
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        
        print(f"\n✅ Total: {len(X_all)} muestras")
        print(f"   Shape: {X_all.shape}")
        
        # Distribución de clases
        unique, counts = np.unique(y_all, return_counts=True)
        print(f"\n📊 Distribución de clases:")
        for label, count in zip(unique, counts):
            signal = config.CLASS_LABELS[label]
            pct = count / len(y_all) * 100
            print(f"   {signal}: {count} ({pct:.1f}%)")
        
        # Split train/val
        split_idx = int(len(X_all) * (1 - config.VALIDATION_SPLIT))
        
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        X_val = X_all[split_idx:]
        y_val = y_all[split_idx:]
        
        print(f"\n✅ Split completado:")
        print(f"   Train: {len(X_train)} muestras")
        print(f"   Val: {len(X_val)} muestras")
        
        return X_train, y_train, X_val, y_val
    
    def train_initial_model(self, symbols=None, timeframe='4h'):
        """Entrena modelo inicial desde cero"""
        print("\n" + "="*60)
        print("🎓 ENTRENAMIENTO INICIAL")
        print("="*60)
        
        # Preparar datos
        X_train, y_train, X_val, y_val = self.prepare_training_data(symbols, timeframe)
        
        # Crear modelo
        input_shape = (X_train.shape[1], X_train.shape[2])  # (lookback, n_features)
        self.model = NeuralTradingModel(input_shape)
        
        # Mostrar resumen
        self.model.get_summary()
        
        # Entrenar
        history = self.model.train(X_train, y_train, X_val, y_val)
        
        # Evaluar
        accuracy = self.model.evaluate(X_val, y_val)
        
        # Guardar
        self.current_version += 1
        self.model.save(self.current_version)
        self.feature_extractor.save_scaler(self.current_version)
        
        # Guardar métricas
        self.save_metrics(self.current_version, {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols or config.DEFAULT_SYMBOLS,
            'timeframe': timeframe
        })
        
        print(f"\n✅ Modelo v{self.current_version} entrenado y guardado")
        
        return self.model
    
    def save_metrics(self, version, metrics):
        """Guarda métricas de un modelo"""
        path = Path(config.MODELS_DIR) / config.METRICS_NAME_FORMAT.format(version=version)
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Métricas guardadas: {path}")
    
    def load_metrics(self, version):
        """Carga métricas de un modelo"""
        path = Path(config.MODELS_DIR) / config.METRICS_NAME_FORMAT.format(version=version)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None


class NeuralStrategy:
    """Interfaz ligera para predicción en tiempo real"""
    
    def __init__(self, version=None):
        """
        Args:
            version: Versión del modelo a cargar (None = última)
        """
        self.cache = DataCache()
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.version = version
        self.input_shape = None
        
        # Cargar modelo
        self.load_model(version)
    
    def load_model(self, version=None):
        """Carga modelo y scaler"""
        learner = ContinuousLearner()
        
        if version is None:
            version = learner.get_latest_version()
        
        if version == 0:
            print("❌ No hay modelos disponibles. Entrena uno primero:")
            print("   python neural_strategy.py --mode train")
            return False
        
        self.version = version
        
        # Cargar scaler
        if not self.feature_extractor.load_scaler(version):
            return False
        
        # Cargar modelo neuronal
        from tensorflow import keras
        model_path = Path(config.MODELS_DIR) / config.MODEL_NAME_FORMAT.format(version=version)
        
        if not model_path.exists():
            print(f"❌ Modelo v{version} no encontrado en {model_path}")
            return False
        
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Modelo v{version} cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
        
        # Obtener input shape del modelo cargado
        self.input_shape = self.model.input_shape[1:]  # (lookback, features)
        
        print(f"✅ Estrategia neuronal v{version} lista")
        return True
    
    def predict_signal(self, X):
        """
        Predice señal con etiqueta
        
        Args:
            X: Features (1, lookback, n_features) o (lookback, n_features)
        
        Returns:
            dict: {'signal': 'BUY'/'SELL'/'HOLD', 'confidence': float, 'probabilities': dict}
        """
        # Asegurar shape correcto
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        
        # Predicción
        probs = self.model.predict(X, verbose=0)[0]
        
        # Clase con mayor probabilidad
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        # Aplicar umbrales de confianza
        signal = config.CLASS_LABELS[predicted_class]
        
        if signal == 'BUY' and confidence < config.MIN_CONFIDENCE_BUY:
            signal = 'HOLD'
        elif signal == 'SELL' and confidence < config.MIN_CONFIDENCE_SELL:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'probabilities': {
                'SELL': float(probs[0]),
                'HOLD': float(probs[1]),
                'BUY': float(probs[2])
            }
        }
    
    def get_signal(self, symbol, timeframe='4h'):
        """
        Obtiene señal de trading para un símbolo
        
        MODO PREDICCIÓN: Solo carga últimas N velas (eficiente)
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe
        
        Returns:
            dict: {'signal': 'BUY'/'SELL'/'HOLD', 'confidence': float, ...}
        """
        # Cargar solo últimas velas necesarias
        df = self.cache.get_data(symbol, timeframe)
        
        if df is None or len(df) < config.LOOKBACK_WINDOW:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': 'Datos insuficientes'
            }
        
        # Tomar solo últimas velas
        df_recent = df.tail(config.LOOKBACK_WINDOW + 50)  # +50 para cálculo de indicadores
        
        # Extraer features
        X = self.feature_extractor.extract_features(df_recent, fit_scaler=False)
        
        # Crear secuencia (solo última)
        X_seq = self.feature_extractor.create_sequences(X)
        
        if len(X_seq) == 0:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': 'No se pudieron crear secuencias'
            }
        
        # Predecir última secuencia
        X_last = X_seq[-1:]
        
        # Verificar que modelo esté cargado
        if self.model is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': 'Modelo no cargado'
            }
        
        # Generar señal
        result = self.predict_signal(X_last)
        result['symbol'] = symbol
        result['timestamp'] = datetime.now().isoformat()
        result['version'] = self.version
        
        return result


# ==================== CLI ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Trading Strategy')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'continuous', 'test'],
                       help='Modo de operación')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Símbolos para entrenar (ej: ETH/USDT BTC/USDT)')
    parser.add_argument('--symbol', type=str, default='ETH/USDT',
                       help='Símbolo para predicción')
    parser.add_argument('--timeframe', type=str, default='4h',
                       help='Timeframe')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Número de épocas')
    parser.add_argument('--version', type=int, default=None,
                       help='Versión del modelo')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("\n🎓 MODO: Entrenamiento Inicial\n")
        learner = ContinuousLearner()
        learner.train_initial_model(args.symbols, args.timeframe)
    
    elif args.mode == 'predict':
        print("\n⚡ MODO: Predicción\n")
        strategy = NeuralStrategy(args.version)
        result = strategy.get_signal(args.symbol, args.timeframe)
        
        print(f"\n{'='*60}")
        print(f"Señal para {result.get('symbol', args.symbol)}")
        print(f"{'='*60}")
        print(f"📊 SEÑAL: {result['signal']}")
        print(f"🎯 Confianza: {result['confidence']:.2%}")
        print(f"\n📈 Probabilidades:")
        for signal, prob in result['probabilities'].items():
            print(f"   {signal}: {prob:.2%}")
        print(f"{'='*60}\n")
    
    elif args.mode == 'test':
        print("\n🧪 MODO: Test de Features\n")
        cache = DataCache()
        df = cache.get_data(args.symbol, args.timeframe)
        
        print(f"✅ Datos cargados: {len(df)} velas")
        
        fe = FeatureExtractor()
        X = fe.extract_features(df, fit_scaler=True)
        
        print(f"✅ Features extraídas: {X.shape}")
        print(f"   Features: {fe.feature_names}")
        
        X_seq = fe.create_sequences(X)
        print(f"✅ Secuencias creadas: {X_seq.shape}")
    
    elif args.mode == 'continuous':
        print("\n🔄 MODO: Aprendizaje Continuo")
        print("⚠️ No implementado aún")
        print("   Este modo ejecutaría reentrenamiento periódico")
