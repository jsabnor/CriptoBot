# Instalación del Servicio Systemd Mejorado

## 🎯 Mejora Implementada

El servicio ahora usa un **script wrapper** (`start_bot.sh`) que:
1. Lee la versión automáticamente del archivo `VERSION`
2. Muestra la versión en los logs al iniciar
3. No requiere editar el servicio manualmente en cada actualización

## 📁 Archivos Creados

### 1. `start_bot.sh` (Script Wrapper)

Script que lee la versión y ejecuta el bot:

```bash
#!/bin/bash
# Script wrapper para el bot que muestra la versión al iniciar

# Leer versión del archivo VERSION
VERSION=$(cat /home/j0s3m4/CriptoBot/VERSION | tr -d '\n\r')

# Mostrar versión
echo "========================================"
echo "Trading Bot v$VERSION"
echo "========================================"

# Ejecutar el bot
exec /home/j0s3m4/CriptoBot/.venv/bin/python /home/j0s3m4/CriptoBot/bot_production.py
```

### 2. `bot.service` (Servicio Actualizado)

Servicio systemd que usa el wrapper:

```ini
[Unit]
Description=Trading Bot Production
After=network.target

[Service]
Type=simple
User=j0s3m4
WorkingDirectory=/home/j0s3m4/CriptoBot
Environment="PATH=/home/j0s3m4/CriptoBot/.venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/j0s3m4/CriptoBot/start_bot.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## 🚀 Instalación en el VPS

### Paso 1: Subir Archivos

```bash
# En el VPS, después de git pull
cd ~/CriptoBot

# Hacer el script ejecutable
chmod +x start_bot.sh

# Verificar que funciona
./start_bot.sh
# Deberías ver: "Trading Bot v1.8.0"
# Presiona Ctrl+C para detener
```

### Paso 2: Actualizar Servicio Systemd

```bash
# Copiar el nuevo archivo de servicio
sudo cp bot.service /etc/systemd/system/bot.service

# Recargar systemd
sudo systemctl daemon-reload

# Reiniciar el servicio
sudo systemctl restart bot.service

# Verificar estado
systemctl status bot.service
```

### Paso 3: Verificar Logs

```bash
# Ver logs en tiempo real
journalctl -u bot.service -f

# Deberías ver algo como:
# ========================================
# Trading Bot v1.8.0
# ========================================
# BOT v1.0 PRODUCTION - MODO: PAPER
```

## ✅ Beneficios

1. **Automático** - La versión se lee del archivo `VERSION`
2. **Sin edición manual** - No necesitas editar el servicio en cada actualización
3. **Visible en logs** - La versión aparece claramente al iniciar
4. **Mantenible** - Un solo lugar para actualizar la versión

## 🔄 Proceso de Actualización Futuro

Ahora, cuando actualices a una nueva versión:

```bash
# 1. Pull cambios
git pull origin main

# 2. Reiniciar servicio (¡eso es todo!)
sudo systemctl restart bot.service

# 3. Verificar versión en logs
journalctl -u bot.service | grep "Trading Bot v"
```

**No necesitas:**
- ❌ Editar `/etc/systemd/system/bot.service`
- ❌ Ejecutar `daemon-reload` (a menos que cambies el servicio)
- ❌ Actualizar la descripción manualmente

## 📊 Comparación

### Antes (v1.4-v1.7)
```bash
# Cada actualización requería:
1. git pull
2. sudo nano /etc/systemd/system/bot.service
3. Cambiar "Description=Trading Bot v1.X"
4. sudo systemctl daemon-reload
5. sudo systemctl restart bot.service
```

### Ahora (v1.8.0+)
```bash
# Solo requiere:
1. git pull
2. sudo systemctl restart bot.service
```

## 🐛 Troubleshooting

### El script no es ejecutable

```bash
chmod +x ~/CriptoBot/start_bot.sh
```

### Permiso denegado

```bash
# Verificar permisos
ls -la ~/CriptoBot/start_bot.sh

# Debería mostrar: -rwxr-xr-x
```

### No aparece la versión en logs

```bash
# Verificar que el archivo VERSION existe
cat ~/CriptoBot/VERSION

# Verificar que el script funciona
~/CriptoBot/start_bot.sh
```

### El servicio no arranca

```bash
# Ver error específico
systemctl status bot.service -l

# Ver logs completos
journalctl -u bot.service -n 50
```

## 📝 Notas Importantes

1. **Primera vez:** Necesitas hacer la instalación completa (Pasos 1-3)
2. **Actualizaciones futuras:** Solo `git pull` y `restart`
3. **El archivo `bot.service` ahora está en el repo** - Se puede versionar
4. **El script `start_bot.sh` también está en el repo** - Versionado automático

## ✨ Resultado

Ahora el servicio systemd es **"self-updating"** en cuanto a la versión. Solo necesitas actualizar el archivo `VERSION` y el servicio mostrará automáticamente la versión correcta en los logs.

¡Mucho más mantenible! 🎉
