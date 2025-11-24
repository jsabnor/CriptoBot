import pandas as pd
import ccxt
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

class DataCache:
    """Maneja caché de datos históricos OHLCV con actualización incremental"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.last_update_file = self.data_dir / '.last_update.json'
    
    def get_cache_path(self, symbol, timeframe='4h'):
        """Ruta del archivo de caché para un símbolo"""
        safe_symbol = symbol.replace('/', '_')
        return self.data_dir / f"{safe_symbol}_{timeframe}.csv"
    
    def download_full_history(self, symbol, timeframe='4h', max_candles=3000):
        """Descarga máximo histórico disponible desde Binance
        
        Args:
            symbol: Par de trading (ej: 'ETH/USDT')
            timeframe: Timeframe (default: '4h')
            max_candles: Máximo de velas a descargar (default: 3000)
        
        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        print(f"📥 Descargando histórico completo de {symbol}...")
        
        all_data = []
        since = None
        candles_per_request = 1000  # Máximo de Binance
        
        # Calcular cuántos requests necesitamos
        num_requests = min(3, (max_candles // candles_per_request) + 1)
        
        for i in range(num_requests):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=since,
                    limit=candles_per_request
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                print(f"  ✓ Bloque {i+1}/{num_requests}: {len(ohlcv)} velas")
                
                # Siguiente bloque desde la última vela + 1ms
                since = ohlcv[-1][0] + 1
                
                # Si recibimos menos de 1000, no hay más datos
                if len(ohlcv) < candles_per_request:
                    print(f"  ℹ No hay más datos históricos disponibles")
                    break
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ⚠️ Error en bloque {i+1}: {e}")
                break
        
        if not all_data:
            print(f"  ❌ No se pudieron descargar datos para {symbol}")
            return None
        
        # Convertir a DataFrame
        df = pd.DataFrame(
            all_data, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Procesar timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Eliminar duplicados y ordenar
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Descargadas {len(df)} velas de {symbol}")
        return df
    
    def save_to_cache(self, symbol, df, timeframe='4h'):
        """Guarda DataFrame en caché CSV"""
        if df is None or len(df) == 0:
            print(f"⚠️ No hay datos para guardar en caché de {symbol}")
            return
        
        cache_path = self.get_cache_path(symbol, timeframe)
        df.to_csv(cache_path, index=False)
        print(f"💾 Guardado en {cache_path} ({len(df)} velas)")
    
    def load_from_cache(self, symbol, timeframe='4h'):
        """Carga datos desde caché CSV"""
        cache_path = self.get_cache_path(symbol, timeframe)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_csv(cache_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"📂 Cargado caché de {symbol}: {len(df)} velas")
            return df
        except Exception as e:
            print(f"❌ Error cargando caché de {symbol}: {e}")
            return None
    
    def update_cache(self, symbol, timeframe='4h'):
        """Actualización incremental del caché
        
        - Si no existe caché: descarga histórico completo
        - Si existe: solo descarga velas nuevas desde la última
        """
        df = self.load_from_cache(symbol, timeframe)
        
        if df is None or len(df) == 0:
            # Primera vez: descargar todo el histórico
            print(f"🆕 Primera descarga para {symbol}")
            df = self.download_full_history(symbol, timeframe)
        else:
            # Actualización incremental
            last_timestamp = df['timestamp'].max()
            since = int(last_timestamp.timestamp() * 1000) + 1
            
            print(f"🔄 Actualizando {symbol} desde {last_timestamp}")
            
            try:
                # Fetch solo velas nuevas
                new_ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=1000
                )
                
                if new_ohlcv and len(new_ohlcv) > 0:
                    new_df = pd.DataFrame(
                        new_ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                    
                    # Append nuevas velas
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    print(f"  ✓ Añadidas {len(new_df)} velas nuevas")
                else:
                    print(f"  ℹ No hay velas nuevas para {symbol}")
            
            except Exception as e:
                print(f"  ⚠️ Error actualizando {symbol}: {e}")
                # Continuar con datos existentes
        
        # Guardar caché actualizado
        if df is not None:
            self.save_to_cache(symbol, df, timeframe)
            self.update_last_update(symbol)
        
        return df
    
    def get_data(self, symbol, timeframe='4h', force_update=False):
        """Obtiene datos OHLCV (desde caché o actualizando si necesario)
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe
            force_update: Fuerza actualización aunque no hayan pasado 4h
        
        Returns:
            DataFrame con datos OHLCV
        """
        if force_update or self.should_update(symbol, timeframe):
            return self.update_cache(symbol, timeframe)
        else:
            df = self.load_from_cache(symbol, timeframe)
            if df is None:
                # Si no hay caché, forzar descarga
                return self.update_cache(symbol, timeframe)
            return df
    
    def should_update(self, symbol, timeframe='4h'):
        """Verifica si necesita actualizar el caché
        
        Actualiza si:
        - No existe el archivo de última actualización
        - No hay registro para este símbolo
        - Han pasado más de 4 horas desde última actualización
        """
        if not self.last_update_file.exists():
            return True
        
        try:
            with open(self.last_update_file, 'r') as f:
                updates = json.load(f)
        except:
            return True
        
        last_update = updates.get(symbol)
        if not last_update:
            return True
        
        try:
            last_dt = datetime.fromisoformat(last_update)
            now = datetime.now()
            
            # Actualizar si pasaron más de 4 horas
            hours_diff = (now - last_dt).total_seconds() / 3600
            return hours_diff >= 4
        except:
            return True
    
    def update_last_update(self, symbol):
        """Registra timestamp de última actualización"""
        updates = {}
        
        if self.last_update_file.exists():
            try:
                with open(self.last_update_file, 'r') as f:
                    updates = json.load(f)
            except:
                updates = {}
        
        updates[symbol] = datetime.now().isoformat()
        
        with open(self.last_update_file, 'w') as f:
            json.dump(updates, f, indent=2)
    
    def get_cache_info(self):
        """Retorna información sobre el caché"""
        info = {}
        
        for csv_file in self.data_dir.glob('*.csv'):
            symbol = csv_file.stem.replace('_', '/')
            df = pd.read_csv(csv_file)
            
            info[symbol] = {
                'candles': len(df),
                'size_kb': csv_file.stat().st_size / 1024,
                'first_date': df['timestamp'].iloc[0] if len(df) > 0 else None,
                'last_date': df['timestamp'].iloc[-1] if len(df) > 0 else None
            }
        
        return info


if __name__ == '__main__':
    # Test del caché
    print("=== Test Data Cache ===\n")
    
    cache = DataCache()
    
    # Test con ETH
    print("\n1. Test descarga inicial ETH/USDT:")
    df = cache.get_data('ETH/USDT', force_update=True)
    
    if df is not None:
        print(f"\nPrimeras 5 velas:")
        print(df.head())
        print(f"\nÚltimas 5 velas:")
        print(df.tail())
    
    # Información del caché
    print("\n2. Información del caché:")
    info = cache.get_cache_info()
    for symbol, data in info.items():
        print(f"\n{symbol}:")
        print(f"  Velas: {data['candles']}")
        print(f"  Tamaño: {data['size_kb']:.1f} KB")
        print(f"  Desde: {data['first_date']}")
        print(f"  Hasta: {data['last_date']}")
