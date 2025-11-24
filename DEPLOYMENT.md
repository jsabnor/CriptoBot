# 🚀 Guía de Despliegue en VPS

Esta guía te ayudará a desplegar el bot de trading en un VPS Ubuntu.

## 📋 Requisitos Previos

- VPS con Ubuntu 20.04 o superior
- Acceso SSH al VPS
- Python 3.8 o superior
- Git instalado

## 🔧 Instalación Inicial

### 1. Conectar al VPS

```bash
ssh usuario@IP_VPS
```

### 2. Instalar Dependencias del Sistema

```bash
# Actualizar paquetes
sudo apt update && sudo apt upgrade -y

# Instalar Python y herramientas
sudo apt install python3 python3-pip python3-venv git screen -y
```

### 3. Clonar el Repositorio

```bash
# Navegar al directorio home
cd ~

# Clonar el proyecto
git clone https://github.com/tu-usuario/CriptoBot.git
cd CriptoBot
```

### 4. Crear Entorno Virtual

```bash
# Crear virtualenv
python3 -m venv .venv

# Activar virtualenv
source .venv/bin/activate

# Verificar que está activo (debe mostrar (.venv) en el prompt)
which python  # Debe mostrar: /home/usuario/CriptoBot/.venv/bin/python
```

### 5. Instalar Dependencias de Python

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# Verificar instalación
pip list | grep -E "(ccxt|pandas|flask)"
```

### 6. Configurar Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar con tus credenciales
nano .env
```

Añade tus claves de Binance:
```bash
BINANCE_API_KEY=tu_clave_api
BINANCE_API_SECRET=tu_secreto_api
TELEGRAM_BOT_TOKEN=tu_token_telegram  # Opcional
TELEGRAM_CHAT_ID=tu_chat_id          # Opcional
TRADING_MODE=paper                    # paper o live
```

Guardar: `Ctrl+O`, `Enter`, `Ctrl+X`

### 7. Inicializar Caché de Datos

```bash
# Descargar datos históricos (toma 2-3 minutos)
python data_cache.py
```

Deberías ver:
```
📥 Descargando HISTORIAL COMPLETO de ETH/USDT...
✅ 18114 velas descargadas
```

## 🚀 Ejecutar el Bot

### Opción 1: Ejecución Directa (Testing)

```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar bot
python bot_production.py
```

**Problema:** Se detiene al cerrar SSH

### Opción 2: Ejecución en Background con Screen (Recomendado)

```bash
# Crear sesión de screen para el bot
screen -S bot
source .venv/bin/activate
python bot_production.py

# Detach (dejar corriendo en background)
# Presiona: Ctrl+A, luego D

# Ver sesiones activas
screen -ls

# Volver a conectarte
screen -r bot

# Matar sesión (si necesitas)
screen -X -S bot quit
```

### Opción 3: Ejecución con tmux

```bash
# Crear sesión tmux
tmux new -s bot
source .venv/bin/activate
python bot_production.py

# Detach: Ctrl+B, luego D
# Reconectar: tmux attach -t bot
```

## 📊 Ejecutar el Dashboard

### Opción 1: Dashboard en Background

```bash
# Crear sesión separada para dashboard
screen -S dashboard
source .venv/bin/activate
python dashboard.py

# Detach: Ctrl+A, luego D
```

### Opción 2: Configurar Firewall

```bash
# Permitir puerto 5000
sudo ufw allow 5000/tcp
sudo ufw status
```

Acceder desde tu navegador: `http://IP_VPS:5000`

### Opción 3: Túnel SSH (Más Seguro)

Desde tu máquina local:

```bash
ssh -L 5000:localhost:5000 usuario@IP_VPS
```

Luego accede a: `http://localhost:5000`

## 🔄 Actualizar el Bot

### Desde el VPS:

```bash
# Navegar al directorio
cd ~/CriptoBot

# Activar virtualenv
source .venv/bin/activate

# Detener bot (si está corriendo)
screen -X -S bot quit

# Actualizar código
git pull origin main

# Instalar nuevas dependencias (si las hay)
pip install -r requirements.txt

# Reiniciar bot
screen -S bot
python bot_production.py
# Ctrl+A, D para detach
```

## 📝 Comandos Útiles

### Gestión de Sesiones Screen

```bash
# Listar sesiones
screen -ls

# Conectar a sesión
screen -r bot          # o dashboard

# Crear nueva sesión
screen -S nombre

# Matar sesión
screen -X -S nombre quit

# Matar todas las sesiones
killall screen
```

### Monitoreo del Bot

```bash
# Ver estado actual
cat bot_state.json | python -m json.tool

# Ver últimos trades
tail -20 trades_production.csv

# Ver logs en tiempo real
screen -r bot
```

### Verificar Caché

```bash
# Ver información del caché
python -c "from data_cache import DataCache; import json; print(json.dumps(DataCache().get_cache_info(), indent=2))"

# Ver última actualización
cat data/.last_update.json
```

## 🔒 Seguridad

### 1. Proteger Archivo .env

```bash
# Permisos solo para el usuario
chmod 600 .env

# Verificar
ls -la .env
# Debe mostrar: -rw------- 1 usuario usuario
```

### 2. Configurar Firewall

```bash
# Habilitar firewall
sudo ufw enable

# Permitir SSH
sudo ufw allow 22/tcp

# Permitir dashboard (opcional)
sudo ufw allow 5000/tcp

# Ver estado
sudo ufw status
```

### 3. Usar Claves SSH

```bash
# Generar clave SSH (en tu máquina local)
ssh-keygen -t ed25519 -C "tu_email@example.com"

# Copiar al VPS
ssh-copy-id usuario@IP_VPS

# Deshabilitar login con contraseña (en VPS)
sudo nano /etc/ssh/sshd_config
# Cambiar: PasswordAuthentication no
sudo systemctl restart sshd
```

## 🐛 Troubleshooting

### El bot no se conecta a Binance

```bash
# Verificar credenciales
cat .env | grep BINANCE

# Probar conexión
python -c "import ccxt; exchange = ccxt.binance(); print(exchange.fetch_ticker('BTC/USDT')['last'])"
```

### Error "No module named 'flask'"

```bash
# Activar virtualenv
source .venv/bin/activate

# Reinstalar dependencias
pip install flask
```

### Dashboard no accesible

```bash
# Verificar que está corriendo
screen -r dashboard

# Verificar firewall
sudo ufw status | grep 5000

# Verificar puerto
netstat -tulpn | grep 5000
```

### Caché desactualizado

```bash
# Forzar actualización
python data_cache.py

# Verificar última actualización
cat data/.last_update.json
```

## 📊 Monitoreo y Mantenimiento

### Revisar Diariamente

1. **Estado del bot**: `cat bot_state.json`
2. **Trades recientes**: `tail trades_production.csv`
3. **Sesiones activas**: `screen -ls`

### Revisar Semanalmente

1. **Actualizar código**: `git pull`
2. **Revisar logs**: Conectar a screen y revisar output
3. **Backup de datos**: Copiar `data/` y `trades_production.csv`

### Backup Automático (Opcional)

```bash
# Crear script de backup
nano ~/backup_bot.sh
```

Contenido:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf ~/backups/bot_backup_$DATE.tar.gz ~/CriptoBot/data ~/CriptoBot/trades_production.csv ~/CriptoBot/bot_state.json
find ~/backups -name "bot_backup_*.tar.gz" -mtime +30 -delete
```

```bash
# Hacer ejecutable
chmod +x ~/backup_bot.sh

# Añadir a crontab (diario a las 2 AM)
crontab -e
# Añadir: 0 2 * * * ~/backup_bot.sh
```

## 🎯 Checklist de Despliegue

- [ ] VPS configurado con Ubuntu
- [ ] Python 3.8+ instalado
- [ ] Git instalado
- [ ] Repositorio clonado
- [ ] Virtualenv creado y activado
- [ ] Dependencias instaladas
- [ ] Archivo .env configurado
- [ ] Caché inicializado
- [ ] Bot ejecutándose en screen
- [ ] Dashboard ejecutándose (opcional)
- [ ] Firewall configurado
- [ ] Backup configurado

## 📞 Soporte

Si tienes problemas:

1. Revisa la sección de Troubleshooting
2. Verifica los logs en la sesión de screen
3. Consulta el CHANGELOG.md para cambios recientes
4. Revisa la documentación en artifacts

---

**¡Listo!** Tu bot está desplegado y corriendo en producción 🚀
