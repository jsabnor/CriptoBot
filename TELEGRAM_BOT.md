# 🤖 Bot de Telegram Interactivo

Sistema de consultas interactivas para los bots de trading mediante comandos de Telegram.

## 📋 Características

- ✅ **Comandos interactivos** con botones inline
- ✅ **Consultas en tiempo real** del estado de los bots
- ✅ **Seguridad por Chat ID** - Solo usuarios autorizados
- ✅ **Independiente** - No interfiere con los bots de trading
- ✅ **Historial de operaciones** personalizable

## 🎯 Comandos Disponibles

### Comandos Básicos

| Comando | Descripción |
|---------|-------------|
| `/start` | Menú principal con botones interactivos |
| `/help` | Lista de comandos disponibles |
| `/status` | Estado de ambos bots (ADX y EMA) |
| `/posiciones` | Posiciones abiertas actuales |
| `/resumen` | Resumen del día actual |
| `/historial` | Últimas 10 operaciones |

### Comandos Avanzados

| Comando | Descripción |
|---------|-------------|
| `/historial adx 7` | Historial del bot ADX (últimos 7 días) |
| `/historial ema 3` | Historial del bot EMA (últimos 3 días) |
| `/adx` | Información específica del bot ADX |
| `/ema` | Información específica del bot EMA |

### Botones Interactivos

Al usar `/start`, aparecerán botones inline para acceso rápido:

```
┌─────────────────────────────────┐
│  📊 Estado  │  💼 Posiciones   │
├─────────────────────────────────┤
│ 📈 Resumen  │  📋 Historial    │
├─────────────────────────────────┤
│ 🤖 Bot ADX  │  📉 Bot EMA      │
├─────────────────────────────────┤
│          ❓ Ayuda              │
└─────────────────────────────────┘
```

## ⚙️ Configuración

### 1. Configurar Variables de Entorno

Edita el archivo `.env` y añade:

```bash
# Bot Token (el mismo que usas para notificaciones)
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz

# Chat ID principal (el mismo de TELEGRAM_CHAT_ID)
TELEGRAM_CHAT_ID=123456789

# Lista de usuarios autorizados (separados por comas)
# Incluye tu Chat ID y el de otras personas autorizadas
TELEGRAM_AUTHORIZED_USERS=123456789,987654321,555666777
```

#### Obtener un Chat ID

**Opción 1 - Usando @userinfobot:**
1. Busca `@userinfobot` en Telegram
2. Envía `/start`
3. El bot te mostrará tu User ID (ese es tu Chat ID)

**Opción 2 - Usando tu propio bot:**
1. Ejecuta el bot interactivo
2. Envía cualquier mensaje a tu bot
3. El bot te mostrará tu Chat ID en el mensaje de "Acceso Denegado"
4. Añade ese ID a `TELEGRAM_AUTHORIZED_USERS`

### 2. Instalar el Servicio

Copia el archivo de servicio a systemd:

```bash
sudo cp telegram_bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable telegram_bot
sudo systemctl start telegram_bot
```

### 3. Verificar que Esté Corriendo

```bash
sudo systemctl status telegram_bot
```

Deberías ver:

```
● telegram_bot.service - Interactive Telegram Bot Handler
   Active: active (running) since ...
   
   🤖 Bot de Telegram Interactivo iniciado
   ✅ Usuarios autorizados: 2
   🚀 Bot iniciado en modo polling...
```

### 4. Ver Logs en Tiempo Real

```bash
sudo journalctl -u telegram_bot -f
```

## 🔒 Seguridad

### Sistema de Autorización

El bot solo responde a usuarios cuyos Chat IDs estén en `TELEGRAM_AUTHORIZED_USERS`.

**Usuarios no autorizados** recibirán:

```
🚫 Acceso Denegado

No tienes autorización para usar este bot.
Tu Chat ID: 999888777
```

### Añadir Usuarios Autorizados

Edita `.env`:

```bash
# Añade más Chat IDs separados por comas
TELEGRAM_AUTHORIZED_USERS=123456789,987654321,555666777
```

Reinicia el servicio:

```bash
sudo systemctl restart telegram_bot
```

## 📊 Ejemplos de Uso

### Consultar Estado General

```
Tú: /status

Bot:
📊 ESTADO DE LOS BOTS

🤖 Bot ADX (Estrategia ADX + ATR)
💰 Equity Total: $203.45
📍 Posiciones: 2/4
📅 Última actualización: 2025-11-27 10:30:00

📉 Bot EMA (Estrategia EMA 15/30)
💰 Equity Total: $198.20
📍 Posiciones: 1/4
📅 Última actualización: 2025-11-27 10:29:45

━━━━━━━━━━━━━━━━━━━━
💼 EQUITY TOTAL: $401.65
```

### Ver Posiciones Abiertas

```
Tú: /posiciones

Bot:
💼 POSICIONES ABIERTAS

🤖 Bot ADX:

🪙 ETH
  └ Entrada: $3245.50
  └ Cantidad: 0.015413
  └ Stop Loss: $3050.20

🪙 XRP
  └ Entrada: $0.5580
  └ Cantidad: 89.605735
  └ Stop Loss: $0.5200

📉 Bot EMA:

🪙 BNB
  └ Entrada: $895.79
  └ Cantidad: 0.026513
  └ Stop Loss: $878.54
```

### Consultar Historial

```
Tú: /historial ema 7

Bot:
📋 HISTORIAL EMA (últimos 7 días)

🔴 26/11 20:05 - BNB
  └ SELL @ $920.45 +$0.65

🟢 26/11 16:00 - BNB
  └ BUY @ $895.79

🔴 25/11 08:30 - SOL
  └ SELL @ $245.20 -$1.20

📊 Mostrando 3 de 3 operaciones
```

## 🛠️ Gestión del Servicio

### Comandos Útiles

```bash
# Iniciar el bot
sudo systemctl start telegram_bot

# Detener el bot
sudo systemctl stop telegram_bot

# Reiniciar el bot
sudo systemctl restart telegram_bot

# Ver estado
sudo systemctl status telegram_bot

# Ver logs
sudo journalctl -u telegram_bot -n 50

# Ver logs en tiempo real
sudo journalctl -u telegram_bot -f
```

### Actualizar el Bot

Si modificas `telegram_bot_handler.py`:

```bash
# Reiniciar para aplicar cambios
sudo systemctl restart telegram_bot

# Verificar que arrancó sin errores
sudo systemctl status telegram_bot
```

## 🔧 Troubleshooting

### Problema: Bot no responde

**Verificar que esté corriendo:**
```bash
sudo systemctl status telegram_bot
```

**Si no está activo:**
```bash
sudo systemctl start telegram_bot
```

### Problema: "Acceso Denegado"

**Solución:**
1. Anota tu Chat ID del mensaje de error
2. Añádelo a `TELEGRAM_AUTHORIZED_USERS` en `.env`
3. Reinicia: `sudo systemctl restart telegram_bot`

### Problema: Comandos no funcionan

**Verificar logs:**
```bash
sudo journalctl -u telegram_bot -n 100
```

Busca errores como:
- `❌ Error leyendo estado de ...` - Archivos JSON no encontrados
- `❌ Error obteniendo updates` - Problemas de conexión

### Problema: No lee datos de los bots

**Verificar archivos de estado:**
```bash
ls -lh ~/CriptoBot/bot_state*.json
ls -lh ~/CriptoBot/trades_*.csv
```

Si faltan archivos, asegúrate de que los bots de trading estén corriendo:
```bash
sudo systemctl status bot
sudo systemctl status bot_ema
```

## 📝 Archivos del Sistema

| Archivo | Ubicación | Descripción |
|---------|-----------|-------------|
| `telegram_bot_handler.py` | `/home/j0s3m4/CriptoBot/` | Código principal del bot |
| `telegram_bot.service` | `/etc/systemd/system/` | Servicio systemd |
| `.env` | `/home/j0s3m4/CriptoBot/` | Configuración (incluye AUTHORIZED_USERS) |
| `bot_state.json` | `/home/j0s3m4/CriptoBot/` | Estado del bot ADX |
| `bot_state_ema.json` | `/home/j0s3m4/CriptoBot/` | Estado del bot EMA |
| `trades_production.csv` | `/home/j0s3m4/CriptoBot/` | Trades del bot ADX |
| `trades_ema.csv` | `/home/j0s3m4/CriptoBot/` | Trades del bot EMA |

## 💡 Consejos de Uso

1. **Añade el bot a favoritos** en Telegram para acceso rápido
2. **Usa los botones** en lugar de escribir comandos manualmente
3. **Comparte acceso** añadiendo Chat IDs de personas de confianza
4. **Monitorea regularmente** el estado con `/status`
5. **Revisa el historial** antes de tomar decisiones manuales

## ⚠️ Limitaciones

- ❌ **No permite** ejecutar trades manualmente
- ❌ **No permite** modificar parámetros de los bots
- ❌ **No permite** detener/iniciar los bots de trading
- ✅ **Solo consulta** información de forma segura

Para operaciones avanzadas, accede al VPS directamente o usa el dashboard web.

---

📱 **Consulta el estado de tus bots en cualquier momento desde Telegram**
