# 📥 Guía de Descarga de Datos Históricos

El script `fetch_data.py` permite descargar datos históricos de Binance para cualquier par y guardarlos en el caché local del bot.

## 🚀 Uso Básico

Ejecuta el script desde la terminal (PowerShell o CMD) en la carpeta del proyecto:

```bash
python fetch_data.py --symbol PAR --start FECHA [OPCIONES]
```

### Argumentos

| Argumento | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--symbol` | Par de trading a descargar | `BTC/USDT`, `ETH/USDT` |
| `--start` | Fecha de inicio (YYYY-MM-DD) | `2023-01-01` |
| `--timeframe` | (Opcional) Intervalo de velas | `4h` (default), `1h`, `15m`, `1d` |

---

## 💡 Ejemplos Comunes

### 1. Descargar historial reciente (recomendado para tests)
Descarga datos de Solana desde Enero 2023 en velas de 4 horas:

```bash
python fetch_data.py --symbol SOL/USDT --start 2023-01-01
```

### 2. Descargar historial largo (recomendado para el bot)
Descarga datos de Ethereum desde 2020 para tener suficiente historial:

```bash
python fetch_data.py --symbol ETH/USDT --start 2020-01-01
```

### 3. Descargar otro timeframe
Descarga datos de Bitcoin en velas de 1 hora:

```bash
python fetch_data.py --symbol BTC/USDT --start 2023-01-01 --timeframe 1h
```

---

## ⚠️ Advertencia Importante sobre el Caché

El sistema de caché del bot (`DataCache`) tiene una **protección de integridad**:

> Si un archivo de caché tiene **menos de 5000 velas** (aprox. 2.3 años en timeframe 4h), el bot asumirá que está incompleto, lo borrará y descargará automáticamente todo el historial disponible desde 2015.

**Recomendación:**
Si planeas usar estos datos con el bot en producción, asegúrate de descargar suficiente historial (al menos 3-4 años para timeframe 4h) para evitar que el bot los sobrescriba.

- **4h**: Mínimo ~2.5 años de datos
- **1h**: Mínimo ~8 meses de datos
- **15m**: Mínimo ~2 meses de datos
