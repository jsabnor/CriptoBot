# Guía para Ejecutar Ambos Bots Simultáneamente

## 🎯 Configuración Dual Bot

Puedes ejecutar tanto el bot ADX como el bot EMA al mismo tiempo, cada uno con su propio capital y estrategia.

## 📋 Configuración del .env

Ambos bots usan el **mismo archivo `.env`**. No necesitas duplicarlo.

```bash
# API Keys (compartidas por ambos bots)
BINANCE_API_KEY=tu_clave_aqui
BINANCE_API_SECRET=tu_secreto_aqui

# Modo de trading (compartido)
TRADING_MODE=paper  # o 'live'

# Configuración de pares (compartida)
SYMBOLS=ETH/USDT,XRP/USDT,BNB/USDT,SOL/USDT
TIMEFRAME=4h

# Capital por par (compartido)
CAPITAL_PER_PAIR=25.0  # 25 EUR por par, por bot = 50 EUR total por par

# Telegram (compartido)
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
DASHBOARD_URL=http://localhost:5000
```

## 💰 Distribución de Capital

### Opción 1: Capital Independiente (Recomendado)

Cada bot opera con su propio capital:

```bash
# En .env
CAPITAL_PER_PAIR=25.0

# Bot ADX: 25 EUR × 4 pares = 100 EUR
# Bot EMA: 25 EUR × 4 pares = 100 EUR
# Total: 200 EUR
```

**Ventajas:**
- ✅ Fácil de gestionar
- ✅ Resultados independientes
- ✅ Puedes detener uno sin afectar al otro

### Opción 2: Capital Compartido

Si quieres que compartan el mismo capital de Binance:

```bash
# En .env
CAPITAL_PER_PAIR=50.0

# Pero solo depositas 200 EUR en Binance
# Cada bot "cree" que tiene 50 EUR/par
# En realidad comparten los 200 EUR
```

**⚠️ Advertencia:** Puede causar errores si ambos intentan usar todo el capital.

## 🔔 Notificaciones de Telegram

Las notificaciones ahora incluyen el prefijo de estrategia:

**Bot ADX:**
```
🟢 [ADX] COMPRA EJECUTADA
🪙 ETH/USDT
...
```

**Bot EMA:**
```
🟢 [EMA] COMPRA EJECUTADA
🪙 ETH/USDT
...
```

Así puedes distinguir fácilmente qué bot generó cada notificación.

## 🚀 Ejecución en Local (Windows)

### Terminal 1 - Bot ADX:
```powershell
cd C:\Users\josem\PROYECTOS\PYTHON\bot
python bot_production.py
```

### Terminal 2 - Bot EMA:
```powershell
cd C:\Users\josem\PROYECTOS\PYTHON\bot
python bot_ema_crossover.py
```

Cada bot:
- ✅ Lee el mismo `.env`
- ✅ Tiene su propio estado (`bot_state.json` vs `bot_state_ema.json`)
- ✅ Registra trades por separado (`trades_production.csv` vs `trades_ema.csv`)
- ✅ Envía notificaciones con su prefijo

## 🖥️ Ejecución en VPS (Ubuntu)

### Instalación de Servicios

```bash
# 1. Copiar archivos de servicio
sudo cp ~/CriptoBot/bot.service /etc/systemd/system/bot.service
sudo cp ~/CriptoBot/bot_ema.service /etc/systemd/system/bot_ema.service

# 2. Recargar systemd
sudo systemctl daemon-reload

# 3. Habilitar ambos servicios
sudo systemctl enable bot.service
sudo systemctl enable bot_ema.service

# 4. Iniciar ambos bots
sudo systemctl start bot.service
sudo systemctl start bot_ema.service

# 5. Verificar estado
systemctl status bot.service
systemctl status bot_ema.service
```

### Gestión de Servicios

**Ver logs en tiempo real:**
```bash
# Bot ADX
journalctl -u bot.service -f

# Bot EMA
journalctl -u bot_ema.service -f

# Ambos simultáneamente
journalctl -u bot.service -u bot_ema.service -f
```

**Detener/Reiniciar:**
```bash
# Detener ambos
sudo systemctl stop bot.service bot_ema.service

# Reiniciar ambos
sudo systemctl restart bot.service bot_ema.service

# Detener solo uno
sudo systemctl stop bot_ema.service
```

**Ver estado:**
```bash
# Estado de ambos
systemctl status bot.service bot_ema.service

# Solo uno
systemctl status bot_ema.service
```

## 📊 Dashboard Web Unificado

Puedes monitorear ambos bots desde una única interfaz web:

```bash
# Iniciar dashboard (si no está corriendo)
python dashboard.py
```

**Características:**
- 🔄 **Vista Combinada**: Equity total y distribución.
- 📈 **Vistas Individuales**: Gráficos y trades de cada bot.
- 🆚 **Comparación**: ROI y Win Rate lado a lado.

Acceso: `http://TU_IP_VPS:5000`

📚 Ver [DASHBOARD_DUAL_BOT.md](DASHBOARD_DUAL_BOT.md) para la guía completa.

## 📊 Monitoreo

### Archivos Generados

**Bot ADX:**
- `bot_state.json` - Estado del bot ADX
- `trades_production.csv` - Trades del bot ADX

**Bot EMA:**
- `bot_state_ema.json` - Estado del bot EMA
- `trades_ema.csv` - Trades del bot EMA

### Ver Estado

```bash
# Estado bot ADX
cat bot_state.json | python -m json.tool

# Estado bot EMA
cat bot_state_ema.json | python -m json.tool

# Trades ADX
tail -20 trades_production.csv

# Trades EMA
tail -20 trades_ema.csv
```

### Comparar Resultados

```python
import pandas as pd

# Cargar trades
adx_trades = pd.read_csv('trades_production.csv')
ema_trades = pd.read_csv('trades_ema.csv')

# Comparar ROI
adx_roi = adx_trades[adx_trades['side']=='sell']['pnl'].sum()
ema_roi = ema_trades[ema_trades['side']=='sell']['pnl'].sum()

print(f"ADX ROI: ${adx_roi:.2f}")
print(f"EMA ROI: ${ema_roi:.2f}")
print(f"Total: ${adx_roi + ema_roi:.2f}")
```

## 🎯 Estrategia Recomendada

### Fase 1: Paper Trading (2 meses)

```bash
# En .env
TRADING_MODE=paper
CAPITAL_PER_PAIR=25.0  # 25 EUR por par, por bot

# Ejecutar ambos bots
# Bot ADX: 100 EUR virtual
# Bot EMA: 100 EUR virtual
# Total: 200 EUR virtual
```

**Objetivo:** Validar que EMA funciona tan bien como ADX.

### Fase 2: Live Trading (Después de validar)

**Opción A - Balanceada (Recomendada):**
```bash
CAPITAL_PER_PAIR=25.0

# Depositar 200 EUR en Binance
# Bot ADX: 100 EUR (estable)
# Bot EMA: 100 EUR (crecimiento)
```

**Opción B - Conservadora:**
```bash
# Si EMA no cumple expectativas en paper
# Solo ejecutar bot ADX
sudo systemctl stop bot_ema.service
```

**Opción C - Agresiva:**
```bash
# Si EMA supera expectativas en paper
CAPITAL_PER_PAIR=50.0

# Depositar 400 EUR
# Bot ADX: 200 EUR
# Bot EMA: 200 EUR
```

## 📈 Proyección con Ambos Bots

**Con 200 EUR inicial + 50 EUR/mes (25 EUR a cada bot):**

| Año | Solo ADX | Solo EMA | Ambos (50/50) |
|-----|----------|----------|---------------|
| 5 | 5,855 EUR | 9,000 EUR | **7,428 EUR** |
| 10 | 19,927 EUR | 35,000 EUR | **27,464 EUR** |
| 12 | 30,278 EUR | 60,000 EUR | **45,139 EUR** |
| 14 | 43,385 EUR | - | **~60,000 EUR** ✅ |

**Con ambos bots: ~14 años para 1,000€/mes** (vs 16 con solo ADX)

## ⚠️ Consideraciones Importantes

1. **Recursos del VPS:**
   - Ambos bots consumen CPU/RAM
   - Asegúrate de tener suficientes recursos
   - Mínimo recomendado: 1 GB RAM

2. **API Rate Limits:**
   - Ambos bots usan la misma API key
   - Binance tiene límites de requests
   - Los bots ya tienen `enableRateLimit=True`

3. **Gestión de Capital:**
   - Si usas capital compartido, ten cuidado
   - Mejor usar capital independiente

4. **Notificaciones:**
   - Recibirás el doble de notificaciones
   - Usa los prefijos [ADX] y [EMA] para distinguir

## 🐛 Troubleshooting

**Error: "Insufficient balance"**
- Verifica que tienes suficiente capital en Binance
- Reduce `CAPITAL_PER_PAIR` si es necesario

**Los bots se interfieren:**
- Asegúrate de que cada bot tiene su propio estado
- Verifica que `bot_state.json` y `bot_state_ema.json` son diferentes

**Demasiadas notificaciones:**
- Normal, son 2 bots
- Puedes silenciar uno temporalmente

**Un bot no arranca:**
```bash
# Ver error específico
journalctl -u bot_ema.service -n 50

# Verificar permisos
ls -la ~/CriptoBot/bot_ema_crossover.py

# Probar manualmente
cd ~/CriptoBot
source .venv/bin/activate
python bot_ema_crossover.py
```

## ✅ Checklist de Configuración

- [ ] `.env` configurado con `CAPITAL_PER_PAIR` adecuado
- [ ] Ambos bots probados en local
- [ ] Servicios systemd creados en VPS
- [ ] Servicios habilitados y arrancados
- [ ] Logs verificados sin errores
- [ ] Notificaciones de Telegram funcionando
- [ ] Prefijos [ADX] y [EMA] visibles
- [ ] Estados independientes (`bot_state.json` vs `bot_state_ema.json`)
- [ ] Trades registrándose por separado

---

**¡Listo para ejecutar ambos bots simultáneamente!** 🚀🚀
