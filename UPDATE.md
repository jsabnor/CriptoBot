# 🔄 Guía de Actualización del Bot

Esta guía explica cómo mantener tu bot actualizado con las últimas mejoras desde GitHub.

## 📋 Tabla de Contenidos

1. [Verificar Actualizaciones](#verificar-actualizaciones)
2. [Aplicar Actualizaciones](#aplicar-actualizaciones)
3. [Proceso de Actualización](#proceso-de-actualización)
4. [Rollback (Restaurar Versión Anterior)](#rollback-restaurar-versión-anterior)
5. [FAQ](#faq)

---

## 🔍 Verificar Actualizaciones

Puedes verificar si hay actualizaciones disponibles sin aplicarlas:

```bash
cd ~/CriptoBot
./check_updates.sh
```

**Salida si estás actualizado:**
```
✓ Tu bot está actualizado
Versión actual: v1.0.0
```

**Salida si hay actualizaciones:**
```
⚠ Hay actualizaciones disponibles
Versión local: v1.0.0
Versión remota: v1.1.0

Cambios disponibles:
* abc1234 Mejorar stop loss dinámico
* def5678 Optimizar cálculo de indicadores
* ghi9012 Corregir bug en gestión de posiciones

Para actualizar, ejecuta: ./update.sh
```

---

## ⬆️ Aplicar Actualizaciones

### Método Automático (Recomendado)

```bash
cd ~/CriptoBot
./update.sh
```

El script hará:
1. ✅ Verificar que hay actualizaciones disponibles
2. ✅ Mostrar cambios que se aplicarán
3. ✅ Solicitar tu confirmación
4. ✅ Crear backup automático
5. ✅ Detener el bot temporalmente
6. ✅ Descargar actualizaciones desde GitHub
7. ✅ Preservar tu archivo `.env` y `bot_state.json`
8. ✅ Actualizar dependencias Python si es necesario
9. ✅ Reiniciar el bot
10. ✅ Verificar que el bot arrancó correctamente

---

## 📊 Proceso de Actualización Detallado

### 1. Preparación

Antes de actualizar, asegúrate de:
- Tener acceso al VPS
- Estar en el directorio del bot
- (Opcional) Revisar el estado actual: `sudo systemctl status bot`

### 2. Ejecutar Script de Actualización

```bash
cd ~/CriptoBot
./update.sh
```

### 3. Revisar Cambios

El script mostrará:

```
========================================
ACTUALIZACIONES DISPONIBLES
========================================

Versión actual:  v1.0.0
Versión nueva:   v1.1.0

Commits nuevos:
* abc1234 Mejorar stop loss dinámico
* def5678 Optimizar cálculo de indicadores

Archivos que cambiarán:
M       bot_production.py
M       requirements.txt
A       utils/indicators.py
```

### 4. Confirmar Actualización

```
¿Deseas aplicar esta actualización?
Se creará un backup automático antes de actualizar
Continuar (s/n):
```

Presiona `s` para continuar o `n` para cancelar.

### 5. El Script Hace el Trabajo

Verás mensajes como:

```
========================================
CREANDO BACKUP
========================================

✓ Backup creado exitosamente
ℹ Ubicación: /home/j0s3m4/CriptoBot_backup_20251122_131500

ℹ Deteniendo el bot...
✓ Bot detenido

========================================
APLICANDO ACTUALIZACIÓN
========================================

ℹ Descargando cambios desde GitHub...
✓ Código actualizado

ℹ Restaurando configuración...
✓ Archivo .env restaurado
✓ Archivo bot_state.json restaurado

========================================
VERIFICANDO DEPENDENCIAS
========================================

ℹ No hay cambios en las dependencias

========================================
REINICIANDO BOT
========================================

ℹ Iniciando el bot...
✓ Bot reiniciado exitosamente

========================================
ACTUALIZACIÓN COMPLETADA
========================================

✓ Bot actualizado a versión v1.1.0
```

### 6. Verificar que Funciona

```bash
# Ver estado del bot
sudo systemctl status bot

# Ver logs en tiempo real
sudo journalctl -u bot -f

# Ver estado del trading
cat bot_state.json
```

---

## 🔙 Rollback (Restaurar Versión Anterior)

Si después de actualizar algo no funciona correctamente, puedes restaurar el backup:

### Encontrar el Backup

Los backups se guardan con timestamp:

```bash
ls -la ~/ | grep CriptoBot_backup
```

Verás algo como:
```
drwxr-xr-x 5 j0s3m4 j0s3m4  4096 Nov 22 13:15 CriptoBot_backup_20251122_131500
```

### Restaurar el Backup

```bash
# 1. Detener el bot
sudo systemctl stop bot

# 2. Hacer backup de la versión problemática (por si acaso)
mv ~/CriptoBot ~/CriptoBot_problematic

# 3. Restaurar el backup
cp -r ~/CriptoBot_backup_20251122_131500 ~/CriptoBot

# 4. Reiniciar el bot
sudo systemctl start bot

# 5. Verificar estado
sudo systemctl status bot
```

---

## 📝 Archivos Preservados Durante Actualización

Estos archivos **NUNCA** se sobrescriben durante una actualización:

| Archivo | Descripción |
|---------|-------------|
| `.env` | Tus claves API y configuración |
| `bot_state.json` | Estado actual del bot (posiciones, equity) |
| `trades_production.csv` | Historial de operaciones |

Están protegidos automáticamente por el script de actualización.

---

## ❓ FAQ

### ¿Con qué frecuencia debo actualizar?

Recomendamos:
- **Verificar actualizaciones**: 1 vez por semana (`./check_updates.sh`)
- **Aplicar actualizaciones**: Cuando haya mejoras importantes o correcciones de bugs

### ¿Se perderá mi configuración?

No. El script preserva automáticamente:
- Archivo `.env` (claves API)
- `bot_state.json` (estado del bot)
- `trades_production.csv` (historial)

### ¿Qué pasa si la actualización falla?

El script crea un backup automático antes de actualizar. Si algo falla:
1. El backup está disponible en `~/CriptoBot_backup_YYYYMMDD_HHMMSS`
2. Puedes restaurarlo siguiendo las instrucciones de [Rollback](#rollback-restaurar-versión-anterior)

### ¿El bot se detendrá durante la actualización?

Sí, temporalmente (1-2 minutos):
1. El bot se detiene antes de aplicar cambios
2. Se aplica la actualización
3. El bot se reinicia automáticamente

Durante este tiempo no se ejecutarán trades.

### ¿Necesito volver a configurar las claves API?

No. Tu archivo `.env` se preserva automáticamente.

### ¿Puedo actualizar manualmente con git pull?

Sí, pero **no es recomendado** porque:
- No crea backup automático
- No preserva archivos de configuración
- No reinicia el bot correctamente
- No actualiza dependencias si es necesario

Es mejor usar el script `./update.sh`.

### ¿Qué pasa si no he configurado git?

El script `./update.sh` detectará esto y te ofrecerá configurar el repositorio automáticamente.

### ¿Cómo sé qué versión tengo instalada?

```bash
cat ~/CriptoBot/VERSION
```

### ¿Puedo revertir a una versión específica?

Sí, manualmente:

```bash
cd ~/CriptoBot
git fetch --all
git checkout tags/v1.0.0  # Reemplaza con la versión deseada
sudo systemctl restart bot
```

**Importante**: Esto no preserva automáticamente tu configuración, hazlo con cuidado.

---

## 🔔 Notificaciones de Actualizaciones

### Verificación Manual

```bash
# Añade esto a tu rutina semanal
cd ~/CriptoBot
./check_updates.sh
```

### Verificación Automática (Opcional)

Puedes configurar un cron job para recibir notificaciones:

```bash
# Editar crontab
crontab -e

# Añadir línea (ejecuta check_updates.sh cada lunes a las 9 AM)
0 9 * * 1 cd /home/j0s3m4/CriptoBot && ./check_updates.sh
```

---

## 📞 Comandos Rápidos

| Acción | Comando |
|--------|---------|
| Verificar actualizaciones | `./check_updates.sh` |
| Aplicar actualizaciones | `./update.sh` |
| Ver versión actual | `cat VERSION` |
| Ver historial de git | `git log --oneline -10` |
| Ver estado del bot | `sudo systemctl status bot` |
| Ver logs del bot | `sudo journalctl -u bot -f` |

---

## ⚠️ Recomendaciones

1. ✅ **Verifica actualizaciones regularmente** - Al menos 1 vez por semana
2. ✅ **Lee los cambios antes de actualizar** - Entiende qué se modificará
3. ✅ **Actualiza en horarios de baja actividad** - Evita periodos de alta volatilidad
4. ✅ **Mantén backups adicionales** - Copia manual de `.env` en lugar seguro
5. ✅ **Monitorea después de actualizar** - Revisa logs durante las primeras horas

---

## 🔗 Enlaces Útiles

- [Repositorio GitHub](https://github.com/jsabnor/CriptoBot)
- [INSTALL.md](INSTALL.md) - Guía de instalación
- [README.md](README.md) - Descripción del proyecto

---

🔄 **Mantén tu bot actualizado para obtener las mejores mejoras y correcciones de bugs**
