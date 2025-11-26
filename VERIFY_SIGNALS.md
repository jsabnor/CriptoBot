# Herramienta de Verificación de Señales (`verify_signals.py`)

Esta herramienta permite simular la lógica de los bots sobre datos históricos recientes para verificar si se deberían haber ejecutado operaciones. Es útil para auditar el comportamiento de los bots y entender por qué se tomó (o no) una operación.

## 📋 Descripción

El script `verify_signals.py`:
1.  Carga los datos históricos (caché) de los pares configurados.
2.  Calcula los mismos indicadores que usan los bots (ADX, EMA, RSI, etc.).
3.  Aplica la lógica exacta de entrada de cada estrategia.
4.  Muestra una lista de las señales detectadas en el periodo especificado.

## 🚀 Uso

Ejecuta el script desde la terminal:

```bash
python verify_signals.py
```

### Opciones Disponibles

| Argumento | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `--days` | Número de días hacia atrás a analizar (por defecto: 2) | `python verify_signals.py --days 5` |
| `--symbol` | Analizar un solo par específico | `python verify_signals.py --symbol ETH/USDT` |

### Ejemplos

**Verificar las últimas 24 horas:**
```bash
python verify_signals.py --days 1
```

**Verificar solo BNB/USDT en la última semana:**
```bash
python verify_signals.py --symbol BNB/USDT --days 7
```

## 📊 Interpretación de Resultados

El script mostrará una tabla detallada con todas las operaciones simuladas:

```text
🔍 VERIFICADOR DE SEÑALES Y OPERACIONES
==================================================
📅 Desde: 2025-11-25 11:30
⏱️ Timeframe: 4h
==================================================

Analizando ETH/USDT...
Analizando BNB/USDT...

====================================================================================================
ESTRATEGIA | SÍMBOLO    | TIPO     | ENTRADA          | SALIDA           | PRECIO ENT.  | PNL %    | RAZÓN
----------------------------------------------------------------------------------------------------
EMA        | ETH/USDT   | SELL     | 11-26 00:00      | Abierta          | $2959.73     | +1.20%   | En Curso
ADX        | BNB/USDT   | SELL     | 11-25 16:00      | 11-25 20:00      | $650.00      | -0.50%   | SL (MA50)
====================================================================================================
```

- **ESTRATEGIA**: Qué bot ejecutó la operación (ADX o EMA).
- **TIPO**: Estado de la operación (OPEN/SELL).
- **ENTRADA/SALIDA**: Fecha y hora de entrada y salida.
- **PNL %**: Resultado porcentual de la operación.
- **RAZÓN**: Por qué se cerró la operación (TP, SL, Cruce, etc.) o si sigue "En Curso".

## 🛠️ Solución de Problemas

- **Si el script dice "No hay datos":** Asegúrate de que el bot haya corrido al menos una vez para descargar el caché, o espera a que se actualice.
- **Si los resultados no coinciden con el Dashboard:** Recuerda que el Dashboard muestra operaciones *reales* ejecutadas. Si el bot estaba apagado o hubo un error de conexión en el momento de la señal, la operación no aparecerá en el Dashboard pero sí en este verificador.
