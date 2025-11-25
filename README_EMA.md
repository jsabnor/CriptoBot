# Bot EMA Crossover - Guía Rápida

## 🎯 Estrategia

**EMA Crossover Optimizada (15/30)**

### Indicadores
- EMA Rápida: 15 períodos
- EMA Lenta: 30 períodos
- ATR: 14 períodos (para stop loss)

### Señales
- **Compra**: EMA 15 cruza EMA 30 al alza
- **Venta**: EMA 15 cruza EMA 30 a la baja
- **Stop Loss**: 2 × ATR por debajo del precio de entrada

### Gestión de Riesgo
- Riesgo por trade: 2% del capital
- Comisión: 0.1%
- Timeframe: 4h

## 📊 Resultados del Backtest (2020-2025)

| Métrica | Valor |
|---------|-------|
| ROI Total | +426% |
| Win Rate | 27.5% |
| Drawdown Máximo | -24.2% |
| Score | 4.85 |
| Sharpe Ratio | 0.14 |

**Comparación con estrategia ADX:**
- ROI: 2.7x mejor (426% vs 159%)
- Drawdown: Similar (-24% vs -19%)
- Tiempo a meta: 11-12 años vs 16 años

## 🚀 Uso

### Configuración

1. **Copiar variables de entorno:**
```bash
# El bot usa el mismo .env que bot_production.py
# Solo asegúrate de tener:
TRADING_MODE=paper
CAPITAL_PER_PAIR=50.0
SYMBOLS=ETH/USDT,XRP/USDT,BNB/USDT,SOL/USDT
```

2. **Ejecutar en paper mode:**
```bash
python bot_ema_crossover.py
```

### Archivos Generados

- `bot_state_ema.json` - Estado del bot (equity, posiciones)
- `trades_ema.csv` - Historial de trades

### Monitoreo

El bot genera los mismos logs que `bot_production.py`:
- Estado cada 4 horas
- Notificaciones de Telegram
- Resumen de equity y ROI

## 📈 Proyección Financiera

**Con 200 EUR inicial + 50 EUR/mes:**

| Año | Capital | Beneficio Mensual |
|-----|---------|-------------------|
| 5 | ~9,000 EUR | ~150 EUR |
| 10 | ~35,000 EUR | ~583 EUR |
| 12 | ~60,000 EUR | ~1,000 EUR ✅ |

**Tiempo a meta: 11-12 años** (vs 16 años con ADX)

## ⚙️ Diferencias vs bot_production.py

| Característica | bot_production.py | bot_ema_crossover.py |
|----------------|-------------------|---------------------|
| Estrategia | ADX + ATR + MA | EMA 15/30 |
| Indicadores | ADX, ATR, MA50, MA200 | EMA15, EMA30, ATR |
| Señal Compra | ADX >20 + MA alcista | EMA15 cruza EMA30 ↑ |
| Señal Venta | MA bajista o TP | EMA15 cruza EMA30 ↓ |
| Stop Loss | ATR × 4 | ATR × 2 |
| Riesgo/Trade | 4% | 2% |
| ROI Esperado | +159% (5 años) | +426% (5 años) |
| Drawdown | -19% | -24% |

## 🎯 Plan de Validación

### Fase 1: Paper Trading (2 meses)
1. Ejecutar bot en paper mode
2. Monitorear trades diarios
3. Comparar con backtest
4. Validar win rate y drawdown

### Fase 2: Decisión (Enero 2026)
- Si ROI >0% y DD <-30% → Pasar a live
- Si resultados no coinciden → Ajustar o descartar

### Fase 3: Live Trading (Febrero 2026)
- Empezar con 200 EUR
- Añadir 50 EUR/mes
- Meta: 1,000 EUR/mes en 2037

## 🔄 Ejecutar Ambos Bots

Puedes ejecutar ambos bots simultáneamente para comparar:

**Terminal 1:**
```bash
python bot_production.py
```

**Terminal 2:**
```bash
python bot_ema_crossover.py
```

Cada uno tendrá su propio:
- Estado (`bot_state.json` vs `bot_state_ema.json`)
- Trades (`trades_production.csv` vs `trades_ema.csv`)
- Capital independiente

## 📝 Notas Importantes

1. **Es más agresivo**: Riesgo 2% vs 4% pero con stop loss más ajustado
2. **Menos trades**: Solo opera en cruces de EMAs
3. **Win rate bajo**: 27.5% es normal, las ganancias vienen de pocos trades grandes
4. **Paciencia**: Puede pasar semanas sin trades
5. **Drawdown**: Espera caídas de hasta -24%

## 🐛 Troubleshooting

**No genera trades:**
- Normal, la estrategia es selectiva
- Verifica que hay cruces de EMAs en TradingView

**Drawdown muy alto:**
- Si supera -30%, revisar configuración
- Considerar reducir riesgo a 1.5%

**Resultados diferentes al backtest:**
- Normal, el mercado cambia
- Dar al menos 2 meses para validar

## 🎊 Ventajas de Esta Estrategia

1. ✅ **Simple**: Solo 2 EMAs
2. ✅ **ROI alto**: 2.7x mejor que ADX
3. ✅ **Drawdown controlado**: -24% vs -53% original
4. ✅ **Probada**: Backtest en 5 años de datos
5. ✅ **Rápida**: 11-12 años vs 16 años

## 📞 Soporte

Si tienes dudas:
1. Revisa los logs del bot
2. Compara con backtest
3. Verifica configuración en `.env`
4. Consulta `strategy_tester.py` para re-validar

---

**¡Listo para empezar el viaje a 1,000 EUR/mes en 12 años!** 🚀
