# 🧠 Bot de Trading Neuronal (v1.1.0)

El **Bot Neuronal** utiliza modelos de Deep Learning (CNN-LSTM) para predecir movimientos de precio y generar señales de compra/venta. Funciona en paralelo con los bots ADX y EMA.

## 📋 Características

*   **Modelo Híbrido**: CNN (patrones visuales) + LSTM (secuencias temporales).
*   **Aprendizaje Continuo**: Capacidad de reentrenarse con nuevos datos.
*   **Gestión de Riesgo**: Stop Loss dinámico y validación de confianza.
*   **Integración Total**: Controlado vía Telegram y `update.sh`.

## 🚀 Inicio Rápido

### 1. Requisitos Previos
Asegúrate de tener modelos entrenados en la carpeta `models/`. Si no, entrena uno:
```bash
python neural_strategy.py --mode train
```

### 2. Ejecución Manual
```bash
python bot_neural.py
```

### 3. Ejecución como Servicio (Recomendado)
El bot se gestiona automáticamente con systemd:
```bash
sudo systemctl start bot_neural
sudo systemctl status bot_neural
```

## 🛠️ Comandos de Gestión

### Ver Logs
```bash
sudo journalctl -u bot_neural -f
```

### Ver Estado
```bash
cat bot_state_neural.json
```

### Reiniciar
```bash
sudo systemctl restart bot_neural
```

## 📊 Estrategia

El bot analiza velas de **4 horas** y utiliza una ventana de contexto (`LOOKBACK_WINDOW`) de 60 velas.

1.  **Extracción de Features**: Calcula RSI, MACD, Bandas de Bollinger, ADX, y retornos logarítmicos.
2.  **Predicción**: El modelo asigna probabilidades a 3 clases: `BUY`, `SELL`, `HOLD`.
3.  **Filtrado**: Solo opera si la confianza supera el umbral configurado (ej. > 60%).

## 📱 Telegram

El bot neural está integrado en el bot interactivo:
*   `/status`: Muestra equity y estado del bot neural.
*   `/posiciones`: Muestra operaciones abiertas con su P&L y confianza del modelo.

## 🔄 Ciclo de Actualización

El script `update.sh` maneja automáticamente:
1.  Parada segura del bot.
2.  Backup del estado (`bot_state_neural.json`).
3.  Actualización de código.
4.  Restauración de estado y reinicio.
