  - Duración de trades en formato legible (Xh Ym, Xd Yh)
  
- **Telegram: Botones interactivos**
  - Acceso directo al dashboard desde notificaciones
  - Links a TradingView para análisis rápido
  - Botones contextuales según tipo de notificación

- **Telegram: Nuevos tipos de notificaciones**
  - `notify_milestone()` - Celebración de logros (ROI +10%, 100 trades, rachas)
  - `notify_risk_alert()` - Alertas de situaciones de riesgo
  - `notify_strong_signal()` - Señales fuertes detectadas sin posición
  - `notify_daily_summary()` - Resumen diario automático
  - `notify_weekly_summary()` - Resumen semanal con top performers

- **Bot: Tracking de duración de trades**
  - Timestamp de entrada en posiciones
  - Cálculo automático de duración
  - Formato legible en notificaciones

### Mejorado
- Mensajes de Telegram más informativos y profesionales
- Mejor experiencia de usuario con acceso rápido a herramientas
- Sistema de notificaciones más completo y motivacional

### Configuración
- Nueva variable `DASHBOARD_URL` en `.env` para botones de Telegram

## [1.7.0] - 2025-11-24

### Añadido
- **Dashboard: Vela actual en tiempo real**
  - El dashboard ahora muestra la vela actual (en progreso) con estilo semi-transparente y borde punteado
  - La vela actual se actualiza automáticamente cada 30 segundos
  - Diferenciación visual clara entre velas cerradas y vela en progreso
  - Etiqueta "Actual (en progreso)" en la leyenda del gráfico

### Cambiado
- **Intervalo de actualización de caché: 4 horas → 5 minutos**
  - El caché de datos históricos ahora se actualiza cada 5 minutos en lugar de cada 4 horas
  - Proporciona datos casi en tiempo real para el dashboard
  - Uso mínimo de API (<0.2% del límite de Binance)
  - Todos los pares se mantienen sincronizados automáticamente

### Mejorado
- Mejor experiencia de usuario en el dashboard con datos más frescos
- Sincronización automática de todos los pares de trading
- Documentación actualizada con nuevos comportamientos del sistema

## [1.6.1] - 2025-11-24

### Corregido
- Versionado del proyecto

## [1.6.0] - 2025-11-23

### Añadido
- Optimizador de estrategia v2 con grid search
- Centralización de configuraciones en `config.py`
- Sistema de caché de datos mejorado

### Cambiado
- Refactorización del sistema de configuración
- Mejoras en el backtesting multi-activo

## [1.5.0] - 2025-11-22

### Añadido
- Dashboard web con Flask
- Visualización de gráficos con Plotly
- Indicadores ADX en tiempo real
- Tabla de trades recientes

### Mejorado
- Sistema de notificaciones por Telegram
- Gestión de posiciones y equity

## [1.0.0] - 2025-11-20

### Añadido
- Bot de trading v1.0 production
- Estrategia basada en ADX, ATR y Moving Averages
- Soporte para timeframe 4h
- Gestión de riesgo con stop loss y take profit
- Modo paper trading y live trading
- Sistema de logging de trades

---

## Tipos de Cambios

- **Añadido**: para nuevas funcionalidades
- **Cambiado**: para cambios en funcionalidades existentes
- **Obsoleto**: para funcionalidades que serán eliminadas
- **Eliminado**: para funcionalidades eliminadas
- **Corregido**: para corrección de bugs
- **Seguridad**: en caso de vulnerabilidades
- **Mejorado**: para mejoras de rendimiento o UX
