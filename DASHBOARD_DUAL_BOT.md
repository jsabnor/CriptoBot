# 📊 Dashboard Dual Bot - Guía de Usuario

## 🎯 Descripción General

El **Dashboard Dual Bot** es una interfaz web unificada diseñada para monitorear y gestionar simultáneamente dos bots de trading:
1.  **Bot ADX**: Estrategia de tendencia (ADX + ATR).
2.  **Bot EMA**: Estrategia de cruce de medias (EMA 15/30).

Esta herramienta proporciona visualización en tiempo real, métricas de rendimiento y comparación directa entre ambas estrategias.

## 🚀 Inicio Rápido

El dashboard se inicia automáticamente con el script `dashboard.py`.

```bash
# Iniciar dashboard
python dashboard.py
```

Acceso:
- **Local**: [http://localhost:5000](http://localhost:5000)
- **Remoto (VPS)**: `http://TU_IP_VPS:5000`

## 🖥️ Vistas del Dashboard

El dashboard cuenta con 4 vistas principales accesibles mediante pestañas:

### 1. 🔄 Vista Combinada (Inicio)
Ofrece una visión global del portafolio.
- **Equity Total**: Suma del capital de ambos bots.
- **Distribución**: Gráfico de anillo mostrando el % de capital en cada bot.
- **ROI Global**: Retorno de inversión ponderado.
- **Estado General**: Resumen rápido de posiciones abiertas.

### 2. 📈 Vista Bot ADX
Detalles específicos para la estrategia ADX.
- **Gráfico Principal**: Velas de 4h + **MA 50** (Línea Naranja).
- **Indicadores**:
  - **ADX** (Línea Roja) en eje secundario derecho.
  - **+DI** (Verde punteada) y **-DI** (Roja punteada) - *Ocultos por defecto, clic en leyenda para ver*.
  - **Threshold** (Línea blanca punteada en 25).
- **Métricas**: Equity, ROI, Posiciones abiertas.
- **Tabla de Trades**: Historial de operaciones del bot ADX.

### 3. 📉 Vista Bot EMA
Detalles específicos para la estrategia EMA.
- **Gráfico Principal**: Velas de 4h.
- **Indicadores**:
  - **EMA 15** (Línea Azul) - Rápida.
  - **EMA 30** (Línea Morada) - Lenta.
- **Métricas**: Equity, ROI, Posiciones abiertas.
- **Tabla de Trades**: Historial de operaciones del bot EMA.

### 4. 🆚 Vista Comparación
Herramientas para analizar qué estrategia funciona mejor.
- **Comparativa de ROI**: Gráfico de barras lado a lado.
- **Comparativa de Win Rate**: Tasa de acierto de cada bot.
- **Tabla Detallada**:
  - Equity, ROI, Trades Totales, Wins/Losses, PnL Total.
  - Columna de **Diferencia** para ver rápidamente quién lidera.

## ✨ Características Clave

- **Zoom Inteligente**: Los gráficos muestran por defecto las últimas **50 velas** para mayor claridad. Puedes hacer scroll para ver el historial.
- **Eje de Tiempo Mejorado**: Formato de fecha claro (`Día/Mes Hora:Min`) con ajuste automático para evitar textos cortados.
- **Actualización Automática**: Los datos se refrescan cada **30 segundos** sin recargar la página.
- **Vela en Progreso**: Visualización de la vela actual (borde punteado) que aún no ha cerrado.
- **Indicadores Específicos**: Cada gráfico muestra solo lo relevante para su estrategia.

## 🛠️ Solución de Problemas

### El dashboard no carga
- Verifica que el proceso esté corriendo: `ps aux | grep python`
- Asegúrate de que el puerto 5000 esté abierto en el firewall del VPS: `sudo ufw allow 5000`

### Los datos no se actualizan
- El dashboard depende de los archivos `bot_state.json` y `bot_state_ema.json`. Verifica que los bots estén escribiendo en estos archivos.
- Revisa la hora de "Última actualización" en la esquina superior derecha.

### Gráficos vacíos
- Si es la primera vez que inicias, puede tardar unos minutos en generarse el caché de datos históricos.
- Ejecuta `python data_cache.py` para forzar una actualización del caché.
