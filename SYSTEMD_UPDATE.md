# Guía de Actualización del Servicio Systemd

## Problema

El archivo de servicio systemd (`/etc/systemd/system/bot.service`) tiene la versión hardcodeada en la descripción:

```ini
[Unit]
Description=Trading Bot v1.4
```

Cuando actualizas el bot, la descripción del servicio no se actualiza automáticamente.

## Solución

### Opción 1: Actualizar Manualmente (Recomendado)

Cada vez que actualices el bot a una nueva versión:

```bash
# 1. Editar el servicio
sudo nano /etc/systemd/system/bot.service

# 2. Cambiar la línea Description
# De: Description=Trading Bot v1.4
# A:  Description=Trading Bot v1.8.0

# 3. Recargar systemd
sudo systemctl daemon-reload

# 4. Verificar
systemctl status bot.service
```

### Opción 2: Descripción Genérica (Sin Versión)

Cambiar la descripción para que no incluya la versión:

```bash
# Editar servicio
sudo nano /etc/systemd/system/bot.service
```

Cambiar a:
```ini
[Unit]
Description=Trading Bot Production
After=network.target
```

Luego:
```bash
sudo systemctl daemon-reload
```

### Opción 3: Script de Actualización Automática

Crear un script que actualice automáticamente:

```bash
# Crear script
nano ~/update_bot_service.sh
```

Contenido:
```bash
#!/bin/bash

# Leer versión del archivo VERSION
VERSION=$(cat ~/CriptoBot/VERSION | tr -d '\n\r')

# Actualizar descripción del servicio
sudo sed -i "s/Description=Trading Bot v.*/Description=Trading Bot v$VERSION/" /etc/systemd/system/bot.service

# Recargar systemd
sudo systemctl daemon-reload

echo "✅ Servicio actualizado a v$VERSION"
systemctl status bot.service | head -5
```

Hacer ejecutable:
```bash
chmod +x ~/update_bot_service.sh
```

Usar después de cada actualización:
```bash
./update_bot_service.sh
```

## Proceso Completo de Actualización

### En el VPS:

```bash
# 1. Navegar al directorio
cd ~/CriptoBot

# 2. Activar virtualenv
source .venv/bin/activate

# 3. Detener bot
sudo systemctl stop bot.service

# 4. Actualizar código
git pull origin main

# 5. Instalar nuevas dependencias (si las hay)
pip install -r requirements.txt

# 6. Actualizar .env si es necesario
# Añadir DASHBOARD_URL=http://tu-vps-ip:5000
nano .env

# 7. Actualizar versión del servicio
sudo nano /etc/systemd/system/bot.service
# Cambiar: Description=Trading Bot v1.8.0

# 8. Recargar systemd
sudo systemctl daemon-reload

# 9. Reiniciar bot
sudo systemctl start bot.service

# 10. Verificar estado
systemctl status bot.service
journalctl -u bot.service -f
```

## Verificación Post-Actualización

```bash
# Ver versión en logs
journalctl -u bot.service | grep "BOT v"

# Ver estado del servicio
systemctl status bot.service

# Ver logs en tiempo real
journalctl -u bot.service -f

# Verificar que está corriendo
ps aux | grep bot_production
```

## Archivo de Servicio Actualizado v1.8.0

```ini
# /etc/systemd/system/bot.service
[Unit]
Description=Trading Bot v1.8.0
After=network.target

[Service]
Type=simple
User=j0s3m4
WorkingDirectory=/home/j0s3m4/CriptoBot
Environment="PATH=/home/j0s3m4/CriptoBot/.venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/j0s3m4/CriptoBot/.venv/bin/python /home/j0s3m4/CriptoBot/bot_production.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Notas Importantes

1. **La versión en el servicio es solo cosmética** - No afecta el funcionamiento del bot
2. **El bot muestra su versión en los logs** al iniciar
3. **Puedes verificar la versión** con: `cat ~/CriptoBot/VERSION`
4. **No olvides actualizar** el archivo `.env` con las nuevas configuraciones

## Troubleshooting

### El servicio no se actualiza

```bash
# Forzar recarga completa
sudo systemctl daemon-reload
sudo systemctl restart bot.service
```

### Ver versión actual del bot

```bash
# En los logs
journalctl -u bot.service | grep "BOT v" | tail -1

# En el archivo
cat ~/CriptoBot/VERSION
```

### Verificar cambios aplicados

```bash
# Ver configuración del servicio
systemctl cat bot.service

# Ver estado detallado
systemctl status bot.service -l
```
