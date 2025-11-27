import argparse
import pandas as pd
import ccxt
from datetime import datetime, timezone
from pathlib import Path
import time
import os

def fetch_data(symbol, start_date, timeframe='4h', data_dir='data'):
    """
    Descarga datos históricos desde una fecha específica y los guarda en caché.
    """
    print(f"\n📥 Iniciando descarga para {symbol}")
    print(f"📅 Desde: {start_date}")
    print(f"⏰ Timeframe: {timeframe}")
    
    # Configurar exchange
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Convertir fecha inicio a timestamp ms
    dt_start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    since = int(dt_start.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            print(f"  🔄 Descargando desde {datetime.fromtimestamp(since/1000, tz=timezone.utc)}...", end=' ')
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                print("✓ Fin de datos")
                break
            
            all_ohlcv.extend(ohlcv)
            print(f"✓ {len(ohlcv)} velas (Total: {len(all_ohlcv)})")
            
            # Actualizar since para la siguiente iteración
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1
            
            if len(ohlcv) < limit:
                print("  ✓ Alcanzado el presente")
                break
                
            # Rate limit
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break
    
    if not all_ohlcv:
        print("❌ No se descargaron datos.")
        return
    
    # Crear DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Guardar en caché
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    safe_symbol = symbol.replace('/', '_')
    filename = data_path / f"{safe_symbol}_{timeframe}.csv"
    
    # Si el archivo ya existe, preguntar si sobrescribir o combinar
    if filename.exists():
        print(f"\n⚠️ El archivo {filename} ya existe.")
        # Por simplicidad en este script, sobrescribimos si es una descarga manual explícita
        print("💾 Sobrescribiendo archivo de caché...")
    
    df.to_csv(filename, index=False)
    print(f"\n✅ Guardado en {filename}")
    print(f"📊 Total velas: {len(df)}")
    print(f"📅 Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    # Advertencia sobre DataCache
    if len(df) < 5000:
        print("\n⚠️ ADVERTENCIA: DataCache requiere mínimo 5000 velas (~2.3 años en 4h).")
        print("   Si usas el bot, podría borrar este archivo y descargar todo desde 2015.")
        print("   Considera descargar una fecha más antigua si planeas usar el bot con este archivo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descargar datos históricos a caché')
    parser.add_argument('--symbol', type=str, required=True, help='Par de trading (ej: BTC/USDT)')
    parser.add_argument('--start', type=str, required=True, help='Fecha inicio YYYY-MM-DD')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe (default: 4h)')
    
    args = parser.parse_args()
    
    fetch_data(args.symbol, args.start, args.timeframe)
