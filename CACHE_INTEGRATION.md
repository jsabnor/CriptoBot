# Integración de Data Cache en bot_production.py

## Cambios necesarios para integrar el caché de datos

### 1. Añadir import (línea 9, después de TelegramNotifier)
```python
from data_cache import DataCache
```

### 2. Inicializar caché en __init__ (después de línea 104, después de self.telegram)
```python
        # Data cache para históricos  
        self.data_cache = DataCache()
```

### 3. Añadir línea en el print de configuración (después de línea 114)
```python
        print(f"Caché de datos: ✓ Activo")
```

### 4. Reemplazar método fetch_ohlcv COMPLETAMENTE (líneas 121-130 aprox)

BÚSQUEDA:
```python
    def fetch_ohlcv(self, symbol, limit=300):
        """Descarga datos OHLCV recientes."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error descargando {symbol}: {e}")
            return None
```

REEMPLAZO:
```python
    def fetch_ohlcv(self, symbol, limit=None):
        """Descarga datos OHLCV desde caché (con fallback a API)."""
        try:
            # Usar caché primero
            df = self.data_cache.get_data(symbol, self.TIMEFRAME)
            
            if df is None or len(df) < self.LONG_MA_LENGTH + 1:
                print(f"⚠️ Caché insuficiente para {symbol}, usando API...")
                return self._fetch_ohlcv_api(symbol, limit or 500)
            
            # Opcional: limitar a últimas N velas
            if limit:
                df = df.tail(limit)
            
            return df
            
        except Exception as e:
            print(f"❌ Error en caché {symbol}: {e}")
            return self._fetch_ohlcv_api(symbol, limit or 500)
    
    def _fetch_ohlcv_api(self, symbol, limit):
        """Fallback: fetch directo desde Binance API."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error API {symbol}: {e}")
            return None
```

## INSTRUCCIONES RÁPIDAS:

En tu PC:
1. Abre `bot_production.py`
2. Línea 9: Añadir `from data_cache import DataCache`
3. Línea 105 (después de `self.telegram = ...`): Añadir `self.data_cache = DataCache()`
4. Línea 115 (en el bloque de prints): Añadir `print(f"Caché de datos: ✓ Activo")`
5. Buscar el método `fetch_ohlcv` y reemplazarlo completo con el nuevo código arriba

GUARDA y sube a GitHub.
