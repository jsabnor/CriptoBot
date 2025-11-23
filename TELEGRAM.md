# 📱 Configuración de Notificaciones por Telegram

Esta guía te mostrará cómo configurar notific

aciones de Telegram para recibir alertas del bot en tiempo real.

## 📋 Tabla de Contenidos

1. [¿Qué Notificaciones Recibirás?](#qué-notificaciones-recibirás)
2. [Crear un Bot de Telegram](#crear-un-bot-de-telegram)
3. [Obtener tu Chat ID](#obtener-tu-chat-id)
4. [Configurar el Bot](#configurar-el-bot)
5. [Probar las Notificaciones](#probar-las-notificaciones)
6. [Deshabilitar Notificaciones](#deshabilitar-notificaciones)
7. [Troubleshooting](#troubleshooting)

---

## 🔔 ¿Qué Notificaciones Recibirás?

### Inicio del Bot
```
🚀 Bot de Trading Iniciado

📊 Modo: SIMULACIÓN
💰 Capital Total: 200.00 EUR
🪙 Pares: ETH, XRP, BNB, SOL
⏰ Timeframe: 4h

🕐 2025-11-23 14:30:00
```

### Compra Ejecutada
```
📈 COMPRA EJECUTADA

🪙 Par: ETH
💵 Precio: $3245.50
📦 Cantidad: 0.015413
💰 Invertido: $50.05

🛑 Stop Loss: $3050.20 (-6.0%)
🎯 Take Profit: $5192.80 (+60.0%)

⏰ 14:35:22
```

### Venta Ejecutada (Ganancia)
```
💰 VENTA EJECUTADA

🪙 Par: ETH
💵 Precio: $3458.20
📦 Cantidad: 0.015413
📋 Razón: Take Profit

💚 P&L: $3.28 (+6.55%)

⏰ 18:45:22
```

### Venta Ejecutada (Pérdida)
```
🛑 VENTA EJECUTADA

🪙 Par: XRP
💵 Precio: $0.5245
📦 Cantidad: 95.418500
📋 Razón: Stop Loss

💔 P&L: -$2.15 (-4.30%)

⏰ 09:12:48
```

### Ciclo Completado
```
📊 Ciclo Completado

💰 Equity: $203.45
📈 ROI Total: +1.73%
💚 P&L: $3.45
📍 Posiciones: 2/4

⏰ 2025-11-23 16:00:00
```

### Error Crítico
```
❌ ERROR CRÍTICO

Error comprando ETH/USDT: Insufficient balance

⏰ 2025-11-23 10:15:33

⚠️ Revisa los logs del bot
```

---

## 🤖 Crear un Bot de Telegram

### 1. Abrir BotFather

1. Abre Telegram en tu teléfono o PC
2. Busca **@BotFather** (es el bot oficial de Telegram para crear bots)
3. Inicia una conversación con `/start`

### 2. Crear tu Bot

Envía el comando:
```
/newbot
```

BotFather te pedirá:

**1. Nombre del bot** (el nombre que aparecerá)
```
Trading Bot Notifier
```

**2. Username del bot** (debe terminar en 'bot')
```
my_trading_notifier_bot
```

### 3. Guardar el Token

BotFather te dará un **token** como este:
```
123456789:ABCdefGHIjklMNOpqrsTUVwxyz
```

**⚠️ IMPORTANTE:** 
- Guarda este token en un lugar seguro
- No lo compartas con nadie
- Lo necesitarás para el archivo `.env`

---

## 🔢 Obtener tu Chat ID

Hay dos métodos:

### Método 1: Usando @userinfobot (Más Fácil)

1. Busca **@userinfobot** en Telegram
2. Inicia una conversación con `/start`
3. El bot te mostrará tu **User ID** (ejemplo: `123456789`)
4. Este es tu Chat ID ✅

### Método 2: Usando tu Bot y un Script

1. Envía un mensaje cualquiera a tu bot (el que creaste con BotFather)
   - Ejemplo: `Hola`

2. En tu VPS, ejecuta este comando:
   ```bash
   curl https://api.telegram.org/bot<TU_TOKEN>/getUpdates
   ```
   
   Reemplaza `<TU_TOKEN>` con el token que te dio BotFather.

3. Busca en la respuesta:
   ```json
   "chat": {
       "id": 123456789,
       ...
   }
   ```

4. Ese número (`123456789`) es tu Chat ID ✅

---

## ⚙️ Configurar el Bot

### 1. Editar el Archivo .env

En tu VPS:

```bash
cd ~/CriptoBot
nano .env
```

### 2. Añadir la Configuración de Telegram

Al final del archivo `.env`, añade:

```env
# Notificaciones de Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

**Reemplaza:**
- `123456789:ABCdefGHIjklMNOpqrsTUVwxyz` con tu token real
- `123456789` con tu Chat ID real

### 3. Guardar y Salir

- Guarda: `Ctrl+O`, luego `Enter`
- Salir: `Ctrl+X`

### 4. Reiniciar el Bot

```bash
sudo systemctl restart bot
```

---

## ✅ Probar las Notificaciones

### Método 1: Reiniciar el Bot

```bash
sudo systemctl restart bot
```

Deberías recibir la notificación de **"Bot Iniciado"** en Telegram.

### Método 2: Ver los Logs

```bash
sudo journalctl -u bot -f
```

Busca líneas como:
```
Telegram: ✓ Habilitado
```

Si dice "✗ Deshabilitado", revisa que las variables en `.env` estén correctas.

### Método 3: Esperar una Operación

El bot te notificará automáticamente cuando:
- Se ejecute una compra
- Se ejecute una venta
- Se complete un ciclo (cada 4 horas)

---

## 🔕 Deshabilitar Notificaciones

Para deshabilitar temporalmente las notificaciones:

### Opción 1: Eliminar las Variables

Edita `.env`:
```bash
nano .env
```

Deja las variables vacías:
```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### Opción 2: Comentar las Líneas

Añade `#` al inicio:
```env
# TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
# TELEGRAM_CHAT_ID=123456789
```

Luego reinicia:
```bash
sudo systemctl restart bot
```

---

## 🔧 Troubleshooting

### Problema: "Bot no recibe notificaciones"

**Solución 1:** Verificar que iniciaste conversación con el bot
```
1. Busca tu bot en Telegram (el username que le pusiste)
2. Presiona "Iniciar" o envía /start
3. Reinicia el bot
```

**Solución 2:** Verificar variables en .env
```bash
cat .env | grep TELEGRAM
```

Debe mostrar tus configuraciones (no vacías).

**Solución 3:** Verificar que el bot esté habilitado
```bash
sudo journalctl -u bot -n 50 | grep Telegram
```

Debe mostrar: `Telegram: ✓ Habilitado`

### Problema: "Error enviando mensaje a Telegram"

**Causas posibles:**
1. Token incorrecto
2. Chat ID incorrecto
3. Firewall bloqueando conexiones salientes

**Solución:**
```bash
# Test manual desde VPS
curl -X POST "https://api.telegram.org/bot<TU_TOKEN>/sendMessage" \
     -d "chat_id=<TU_CHAT_ID>" \
     -d "text=Test desde VPS"
```

Si funciona, el problema está en el archivo `.env`.

### Problema: "Bot dice 'Telegram: ✗ Deshabilitado'"

**Causas:**
- Variables `TELEGRAM_BOT_TOKEN` o `TELEGRAM_CHAT_ID` vacías
- Variables mal escritas en `.env`

**Solución:**
```bash
# Verificar archivo .env
cat .env

# Debe tener:
TELEGRAM_BOT_TOKEN=algo_aqui
TELEGRAM_CHAT_ID=numeros_aqui
```

### Problema: "Notificaciones llegan con retraso"

Esto es normal. Las notificaciones de Telegram pueden tener hasta 1-2 minutos de retraso, especialmente:
- Notificaciones "silenciosas" (ciclo completado)
- Cuando hay muchas notificaciones seguidas

---

## 📊 Tipos de Notificaciones

| Tipo | Emoji | Cuándo se Envía | Silenciosa |
|------|-------|-----------------|------------|
| Inicio | 🚀 | Al iniciar el bot | No |
| Compra | 📈 | Al ejecutar compra | No |
| Venta (TP) | 💰 | Take profit alcanzado | No |
| Venta (SL) | 🛑 | Stop loss activado | No |
| Venta (MA SL) | ⚠️ | Stop loss por MA | No |
| Venta (Bearish) | 📉 | Señal bajista | No |
| Ciclo Completo | 📊 | Cada 4 horas | Sí |
| Error | ❌ | Error crítico | No |

**Nota:** Las notificaciones "silenciosas" no hacen sonido en tu teléfono.

---

## 💡 Consejos

1. ✅ **Silencia las notificaciones de noche** en la configuración de Telegram
2. ✅ **Crea un grupo** solo para el bot si quieres compartir notificaciones
3. ✅ **Revisa las notificaciones** al menos 1 vez al día
4. ✅ **No compartas tu token** con nadie
5. ✅ **Guarda el token** en un gestor de contraseñas

---

## 🔗 Enlaces Útiles

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [BotFather](https://t.me/BotFather)
- [@userinfobot](https://t.me/userinfobot)

---

📱 **Recibe notificaciones en tiempo real de todas las operaciones del bot**
