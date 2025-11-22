# 📦 Guía de Instalación - Bot de Trading v1.0

Esta guía te mostrará cómo instalar el bot de trading desde cero en un VPS (Ubuntu/Debian).

## 📋 Tabla de Contenidos

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación Rápida (Script Automatizado)](#instalación-rápida-script-automatizado)
3. [Instalación Manual](#instalación-manual)
4. [Configuración de Claves API](#configuración-de-claves-api)
5. [Configuración del Bot](#configuración-del-bot)
6. [Ejecución del Bot](#ejecución-del-bot)
7. [Ejecutar como Servicio (systemd)](#ejecutar-como-servicio-systemd)
8. [Monitoreo y Logs](#monitoreo-y-logs)
9. [Troubleshooting](#troubleshooting)

---

## 📋 Requisitos del Sistema

### Sistema Operativo
- **Ubuntu 20.04+** o **Debian 10+** (recomendado para VPS)
- Acceso root o sudo

### Hardware Mínimo
- **CPU**: 1 core
- **RAM**: 1 GB
- **Disco**: 5 GB libres
- **Conexión a Internet**: Estable (para conexión con Binance API)

### Software
- **Python**: 3.9 o superior
- **pip**: Gestor de paquetes de Python
- **git**: (opcional) para clonar el repositorio

---

## 🚀 Instalación Rápida (Script Automatizado)

Si quieres instalar todo automáticamente, usa el script `install.sh`:

```bash
# 1. Descarga el script de instalación
wget https://raw.githubusercontent.com/tu-usuario/bot/main/install.sh

# O si ya tienes el proyecto:
cd /ruta/a/tu/bot

# 2. Dale permisos de ejecución
chmod +x install.sh

# 3. Ejecuta el script
sudo ./install.sh
```

El script instalará:
- ✅ Python 3.9+ y pip
- ✅ Entorno virtual Python
- ✅ Todas las dependencias (pandas, numpy, ccxt, etc.)
- ✅ Configuración de variables de entorno
- ✅ (Opcional) Servicio systemd

**¡Salta a la sección [Configuración de Claves API](#configuración-de-claves-api) después de ejecutar el script!**

---

## 🔧 Instalación Manual

Si prefieres instalar paso a paso:

### 1. Actualizar el Sistema

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Instalar Python 3.9+ y pip

```bash
# Verificar versión de Python (debe ser 3.9+)
python3 --version

# Si no está instalado o es una versión antigua:
sudo apt install python3 python3-pip python3-venv -y
```

### 3. Instalar Git (opcional)

```bash
sudo apt install git -y
```

### 4. Clonar o Subir el Proyecto

**Opción A: Clonar desde Git**
```bash
cd /opt
sudo git clone https://github.com/tu-usuario/bot.git
cd bot
```

**Opción B: Subir archivos por SFTP/SCP**
```bash
# En tu máquina local:
scp -r /ruta/local/bot usuario@tu-vps:/opt/bot

# En el VPS:
cd /opt/bot
```

### 5. Crear Entorno Virtual

```bash
cd /opt/bot
python3 -m venv .venv
source .venv/bin/activate
```

### 6. Instalar Dependencias Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Las dependencias incluyen:
- `pandas>=1.5.0` - Manipulación de datos
- `numpy>=1.23.0` - Cálculos numéricos
- `ccxt>=4.0.0` - Conexión con exchanges
- `python-dotenv>=1.0.0` - Variables de entorno

---

## 🔑 Configuración de Claves API

### 1. Obtener Claves de Binance

1. Inicia sesión en [Binance](https://www.binance.com)
2. Ve a **Perfil** → **API Management**
3. Crea una nueva API Key:
   - Nombre: `Bot Trading v1.0`
   - IP Restriction: **Habilita** y añade la IP de tu VPS (recomendado)
   - Permisos: **Enable Reading** y **Enable Spot & Margin Trading**
   - ⚠️ **NO habilites** "Enable Withdrawals"
4. Guarda tu **API Key** y **Secret Key** de forma segura

### 2. Configurar Variables de Entorno

```bash
cd /opt/bot

# Copiar plantilla de configuración
cp .env.example .env

# Editar archivo .env
nano .env
```

Contenido del archivo `.env`:

```env
# Claves API de Binance
BINANCE_API_KEY=tu_clave_api_aqui
BINANCE_API_SECRET=tu_secreto_api_aqui

# Modo de trading: 'paper' (simulación) o 'live' (real)
TRADING_MODE=paper

# Capital inicial por par (EUR)
CAPITAL_PER_PAIR=50.0
```

**⚠️ IMPORTANTE:**
- Reemplaza `tu_clave_api_aqui` con tu API Key real
- Reemplaza `tu_secreto_api_aqui` con tu Secret Key real
- **Empieza siempre en modo `paper`** para probar sin riesgo
- Guarda el archivo (`Ctrl+O`, `Enter`, `Ctrl+X` en nano)

### 3. Proteger el Archivo .env

```bash
# Solo el propietario puede leer/escribir
chmod 600 .env

# Verificar permisos
ls -la .env
# Debe mostrar: -rw------- 1 usuario usuario
```

---

## ⚙️ Configuración del Bot

El bot ya viene configurado con valores óptimos basados en backtesting:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Timeframe** | 4h | Intervalo de análisis |
| **Pares** | ETH, XRP, BNB, SOL | Criptomonedas a operar |
| **Capital por par** | 50 EUR | Capital asignado a cada par |
| **Capital total** | 200 EUR | 50 EUR × 4 pares |
| **Riesgo por trade** | 2% | Máximo a arriesgar por operación |
| **ROI esperado** | 8-15% anual | Rendimiento realista |

Si quieres cambiar estos valores, edita `bot_production.py` (no recomendado para principiantes).

---

## 🚀 Ejecución del Bot

### Modo Manual (Testing)

```bash
cd /opt/bot
source .venv/bin/activate
python bot_production.py
```

El bot:
- 🔄 Analizará los 4 pares cada 4 horas
- 📊 Mostrará el estado de posiciones
- 💾 Guardará estado en `bot_state.json`
- 📝 Registrará trades en `trades_production.csv`

**Detener el bot**: Presiona `Ctrl+C`

### Verificar que Funciona

Después de unos segundos, deberías ver:

```
======================================================================
BOT v1.0 PRODUCTION - MODO: PAPER
======================================================================
Timeframe: 4h
Pares: 4
  1. ETH/USDT (Capital: 50.0 EUR)
  2. XRP/USDT (Capital: 50.0 EUR)
  3. BNB/USDT (Capital: 50.0 EUR)
  4. SOL/USDT (Capital: 50.0 EUR)
Capital Total: 200.0 EUR
======================================================================

🔄 CICLO DE TRADING - 2025-11-22 12:00:00
======================================================================

📊 Procesando ETH/USDT...
📊 Procesando XRP/USDT...
...
```

---

## 🔄 Ejecutar como Servicio (systemd)

Para que el bot se ejecute automáticamente en segundo plano:

### 1. Copiar el Archivo de Servicio

```bash
sudo cp bot.service /etc/systemd/system/bot.service
```

### 2. Editar el Archivo de Servicio (si es necesario)

```bash
sudo nano /etc/systemd/system/bot.service
```

Verifica que las rutas sean correctas:
```ini
[Service]
User=tu_usuario
WorkingDirectory=/opt/bot
ExecStart=/opt/bot/.venv/bin/python /opt/bot/bot_production.py
```

Reemplaza `tu_usuario` con tu nombre de usuario real.

### 3. Habilitar e Iniciar el Servicio

```bash
# Recargar configuración de systemd
sudo systemctl daemon-reload

# Habilitar inicio automático
sudo systemctl enable bot

# Iniciar el servicio
sudo systemctl start bot

# Verificar estado
sudo systemctl status bot
```

Deberías ver:
```
● bot.service - Trading Bot v1.0
   Loaded: loaded (/etc/systemd/system/bot.service; enabled)
   Active: active (running) since ...
```

### 4. Gestión del Servicio

```bash
# Detener el bot
sudo systemctl stop bot

# Reiniciar el bot
sudo systemctl restart bot

# Ver logs en tiempo real
sudo journalctl -u bot -f

# Ver logs de las últimas 24 horas
sudo journalctl -u bot --since "24 hours ago"
```

---

## 📊 Monitoreo y Logs

### Estado del Bot

```bash
# Ver estado del bot (si corre como servicio)
sudo systemctl status bot

# Ver archivo de estado (posiciones actuales)
cat /opt/bot/bot_state.json

# Ver historial de trades
cat /opt/bot/trades_production.csv
```

### Logs del Sistema

```bash
# Ver logs en tiempo real
sudo journalctl -u bot -f

# Ver últimos 100 logs
sudo journalctl -u bot -n 100

# Ver logs de hoy
sudo journalctl -u bot --since today
```

### Monitoreo de Rendimiento

```bash
# Ver uso de CPU y RAM
top
# (Busca el proceso 'python')

# Ver conexiones de red
sudo netstat -tulpn | grep python
```

### Archivos Generados

El bot genera estos archivos automáticamente:

| Archivo | Descripción |
|---------|-------------|
| `bot_state.json` | Estado actual (posiciones, equity, timestamp) |
| `trades_production.csv` | Historial de todas las operaciones |
| `.env` | Configuración y claves API (¡NO compartir!) |

---

## 🔧 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'pandas'"

**Solución:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Problema: "binance.exceptions.AuthenticationError"

**Causas posibles:**
1. Claves API incorrectas en `.env`
2. IP del VPS no autorizada en Binance
3. Permisos de API insuficientes

**Solución:**
```bash
# 1. Verificar archivo .env
cat .env

# 2. En Binance, verificar:
#    - Claves API correctas
#    - IP del VPS añadida a whitelist
#    - Permisos: "Reading" y "Spot Trading" habilitados

# 3. Obtener IP del VPS
curl ifconfig.me
```

### Problema: "ERROR: Could not find a version that satisfies the requirement..."

**Solución:**
```bash
# Actualizar pip
pip install --upgrade pip

# Reinstalar dependencias
pip install -r requirements.txt --upgrade
```

### Problema: El servicio no inicia

**Solución:**
```bash
# Ver error específico
sudo journalctl -u bot -n 50

# Verificar permisos
ls -la /opt/bot/bot_production.py

# Verificar que .env existe
ls -la /opt/bot/.env

# Probar manualmente primero
cd /opt/bot
source .venv/bin/activate
python bot_production.py
```

### Problema: "Permission denied" al crear archivos

**Solución:**
```bash
# Dar permisos al directorio
sudo chown -R tu_usuario:tu_usuario /opt/bot

# Verificar permisos
ls -la /opt/bot
```

### Problema: Bot no conecta con Binance

**Solución:**
```bash
# Verificar conectividad
ping api.binance.com

# Verificar firewall
sudo ufw status

# Si es necesario, permitir conexión saliente
sudo ufw allow out 443/tcp
```

### Problema: "Out of memory" o bot se cierra solo

**Causas:** VPS con poca RAM

**Solución:**
```bash
# Crear swap (memoria virtual)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Hacer permanente
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## ⚠️ Recomendaciones de Seguridad

1. ✅ **Nunca compartas tu archivo `.env`**
2. ✅ **Usa restricción de IP en Binance** (whitelist tu VPS)
3. ✅ **Deshabilita withdrawals** en las claves API
4. ✅ **Empieza con modo `paper`** durante 1-2 meses
5. ✅ **Haz backups** de `bot_state.json` y `trades_production.csv`
6. ✅ **Monitorea diariamente** el estado del bot
7. ✅ **Solo invierte lo que puedas perder**

---

## 📞 Próximos Pasos

1. ✅ **Verifica la instalación**: Ejecuta el bot manualmente y confirma que se conecta
2. ✅ **Modo Paper Trading**: Déjalo correr 1-2 meses para observar comportamiento
3. ✅ **Revisa logs diariamente**: Asegúrate de que no hay errores
4. ✅ **Analiza resultados**: Revisa `trades_production.csv` semanalmente
5. ⚠️ **Considera Live Trading**: Solo después de confirmar buenos resultados en paper

---

## 📚 Archivos de Referencia

- [README.md](README.md) - Descripción general del proyecto
- [UPDATE.md](UPDATE.md) - Guía de actualizaciones
- [requirements.txt](requirements.txt) - Lista de dependencias
- [bot_production.py](bot_production.py) - Código principal del bot
- [.env.example](.env.example) - Plantilla de configuración

---

## 🔄 Actualizaciones

Para mantener el bot actualizado con las últimas mejoras y correcciones:

```bash
# Verificar si hay actualizaciones disponibles
./check_updates.sh

# Aplicar actualizaciones automáticamente
./update.sh
```

📚 **Consulta [UPDATE.md](UPDATE.md) para más información sobre el sistema de actualización**

El script de actualización:
- ✅ Crea backup automático
- ✅ Preserva tu configuración (`.env`, `bot_state.json`)
- ✅ Actualiza dependencias si es necesario
- ✅ Reinicia el bot automáticamente

---

**¿Necesitas ayuda?** Revisa la sección [Troubleshooting](#troubleshooting) o consulta los logs del bot.

🤖 **Bot v1.0 Production | ROI esperado 8-15% anual**
