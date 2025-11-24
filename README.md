# 🤖 Bot de Trading v1.0 Production

## 🚀 Inicio Rápido

```bash
# Ejecutar bot en paper trading
python bot_production.py
```

## 📁 Estructura del Proyecto

```
bot/
├── bot_production.py              ⭐ Bot principal (4h, ETH/XRP/BNB/SOL)
├── backtest_multi.py              🧪 Sistema de backtesting
├── generate_dashboard.py          📊 Generador de gráficas
├── backtest_multi_results.csv     📈 Resultados 21 tests
├── roi_comparison_4h.png          📊 Gráfica ROI 4h
├── timeframe_comparison.png       📊 Comparación timeframes
├── roi_heatmap.png                📊 Heatmap completo
├── top10_configs.png              📊 Top 10 configuraciones
├── data/                          📁 Datos históricos OHLCV
└── archive_old_versions/          📦 Versiones antiguas
```

## 📊 Resultados del Backtesting (2020-2025)

| Par | Timeframe | ROI | Trades |
|-----|-----------|-----|--------|
| ETH/USDT | 4h | **+91.4%** | 220 |
| XRP/USDT | 4h | **+86.9%** | 239 |
| BNB/USDT | 4h | **+82.4%** | 239 |
| SOL/USDT | 4h | **+75.6%** | 203 |

**Tasa de éxito**: 95.2% (20/21 tests positivos)

## 💰 ROI Anual Esperado

**Estimación realista**: **8-15% anual**

- Conservador: 8-10%
- Realista: 10-12%
- Optimista: 12-15%

*Mucho mejor que cuentas de ahorro (0.5-2%) o S&P 500 (~10%)*

## ⚙️ Configuración del Bot

```python
Timeframe: 4h
Pares: ETH, XRP, BNB, SOL
Capital: 50 EUR × 4 = 200 EUR
Riesgo: 2% por trade
Modo: Paper Trading (default)
```

## 📚 Documentación Completa

Los siguientes documentos están disponibles en artifacts:

1. **README.md** - Este archivo
2. **guia_uso_bot_production.md** - Guía completa de uso
3. **analisis_multi_activo_multi_tf.md** - Análisis backtesting
4. **resumen_proyecto_completo.md** - Resumen del proyecto

## 🎯 Próximos Pasos

1. **Ver gráficas** - Abre los 4 archivos PNG
2. **Leer guía** - `guia_uso_bot_production.md` en artifacts
3. **Paper trading** - `python bot_production.py`
4. **Monitorear** - Revisar `bot_state.json` diariamente

## 🔄 Actualizaciones

Mantén el bot actualizado con las últimas mejoras:

```bash
# Verificar si hay actualizaciones
./check_updates.sh

# Aplicar actualizaciones
./update.sh
```

📚 Ver [UPDATE.md](UPDATE.md) para más detalles

## 💾 Sistema de Caché

El bot utiliza un sistema de caché inteligente para optimizar llamadas a la API de Binance:

**Características:**
- 📊 **Velas históricas**: Actualizadas cada 5 minutos
- ⚡ **Vela actual**: Obtenida en tiempo real (cada 30s en dashboard)
- 💿 **Almacenamiento**: Archivos CSV en directorio `data/`
- 🎯 **Uso de API**: <0.2% del límite de Binance (totalmente seguro)

**Actualización manual:**
```bash
# Forzar actualización de un par específico
python -c "from data_cache import DataCache; DataCache().get_data('ETH/USDT', '4h', force_update=True)"

# Actualizar todos los pares
python data_cache.py
```

**Beneficios:**
- ✅ Dashboard siempre actualizado (máximo 5 min de retraso)
- ✅ Sincronización automática de todos los pares
- ✅ Sin riesgo de rate limiting
- ✅ Carga rápida desde archivos locales

## 📱 Notificaciones por Telegram

El bot puede enviarte notificaciones en tiempo real de todas las operaciones:

- 📈 **Compras** (con precio, cantidad, SL y TP esperados)
- 📉 **Ventas** (con P&L y ROI)
- 📊 **Estado del bot** (inicio, ciclos, errores)

📚 Ver [TELEGRAM.md](TELEGRAM.md) para configurar notificaciones

## 📊 Dashboard Web

Interfaz web para monitorear el bot con gráficos interactivos en tiempo real:

- 📈 **Gráficos de velas en tiempo real** con vela actual en progreso
- 📊 **Indicadores técnicos**: ADX, MA50, MA200
- 🟢🔴 **Marcadores de compra/venta** en los gráficos
- 💰 **Métricas en tiempo real** (equity, ROI, posiciones)
- 📋 **Historial de trades** con P&L
- 🔄 **Auto-refresh**: Actualización cada 30 segundos

**Características v1.7.0:**
- ✨ **Vela actual visible**: Muestra la vela en progreso con estilo semi-transparente
- ⚡ **Datos frescos**: Caché actualizado cada 5 minutos
- 🎨 **Diferenciación visual**: Vela actual con borde punteado y opacidad 70%

```bash
# Iniciar dashboard
python dashboard.py

# Acceder desde navegador local
http://localhost:5000

# O desde VPS
http://IP_VPS:5000
```

📚 Ver [DASHBOARD.md](DASHBOARD.md) para configuración completa

## ⚠️ Importante

- ✅ Empezar con **paper trading** (no usa dinero real)
- ✅ Mínimo **1-2 meses** antes de live trading
- ✅ Solo usar **capital que puedas perder**
- ⚠️ ROI esperado **8-15% anual** (no 100%)

## 📞 Archivos Auto-Generados

Cuando ejecutes el bot, se crearán:
- `bot_state.json` - Estado actual
- `trades_production.csv` - Log de operaciones

---

*Bot v1.0 Production | Optimizado para 4h | ROI esperado 8-15% anual*
