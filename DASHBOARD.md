# 📊 Dashboard Web del Trading Bot

Interfaz web para monitorear el bot de trading en tiempo real con gráficos interactivos.

## 🎯 Características

- ✅ **Gráficos de velas interactivos** (Plotly)
- ✅ **Indicadores técnicos**: MA50, MA200
- ✅ **Marcadores de trades**: 🟢 Compras | 🔴 Ventas
- ✅ **Métricas en tiempo real**: Equity, ROI, Posiciones
- ✅ **Historial de trades** con P&L
- ✅ **Auto-actualización** cada 30 segundos
- ✅ **Tema oscuro** profesional

---

## 📋 Requisitos Previos

El dashboard lee los datos del bot, por lo que necesitas:

1. ✅ Bot funcionando (PAPER o LIVE)
2. ✅ Archivos `bot_state.json` y `trades_production.csv` generados
3. ✅ Dependencias instaladas (`Flask`, `plotly`)

---

## 🚀 Instalación

### 1. Instalar Dependencias

El script `update.sh` ya actualiza dependencias automáticamente, pero si instalas manualmente:

```bash
cd ~/CriptoBot
source .venv/bin/activate
pip install Flask plotly
```

### 2. Configurar el Servicio

```bash
# Copiar archivo de servicio
sudo cp dashboard.service /etc/systemd/system/

# Recargar systemd
sudo systemctl daemon-reload

# Habilitar inicio automático
sudo systemctl enable dashboard

# Iniciar dashboard
sudo systemctl start dashboard

# Verificar estado
sudo systemctl status dashboard
```

---

## 🌐 Acceso al Dashboard

### Desde el Navegador

```
http://IP_DEL_VPS:5000
```

**Ejemplos:**
- Local: `http://localhost:5000`
- Red local: `http://192.168.1.100:5000`
- VPS público: `http://TU_IP_PUBLICA:5000`

> **⚠️ Firewall**: Asegúrate de que el puerto 5000 esté abierto:
> ```bash
> sudo ufw allow 5000
> ```

---

## 📊 Uso del Dashboard

### Métricas Principales

| Métrica | Descripción |
|---------|-------------|
| 💰 **Total Equity** | Capital total actual |
| 📈 **ROI Total** | Rendimiento desde inicio |
| 📍 **Posiciones** | Posiciones abiertas/total |
| 📊 **Total Trades** | Número de compras realizadas |

### Gráficos de Velas

Cada par (ETH, XRP, BNB, SOL) tiene su propio gráfico mostrando:

- **Velas OHLC** (4h timeframe)
- **MA50** (naranja) - Media móvil 50 períodos
- **MA200** (azul) - Media móvil 200 períodos
- **🟢 Marcadores verdes** - Compras realizadas
- **🔴 Marcadores rojos** - Ventas realizadas

**Interacción:**
- 🔍 **Zoom**: Arrastrar en el gráfico
- 🖱️ **Pan**: Shift + Arrastrar
- ℹ️ **Hover**: Ver detalles de cada vela
- 🏠 **Reset**: Doble click

### Tabla de Trades

Muestra los últimos 20 trades con:
- Timestamp
- Par (ETH, XRP, BNB, SOL)
- Tipo (BUY/SELL)
- Precio de ejecución
- Cantidad
- Razón de venta (TP, SL, MA_SL, bearish)
- **P&L** (verde = ganancia, rojo = pérdida)

---

## 🔧 Comandos Útiles

```bash
# Ver estado del dashboard
sudo systemctl status dashboard

# Ver logs en tiempo real
sudo journalctl -u dashboard -f

# Reiniciar dashboard
sudo systemctl restart dashboard

# Detener dashboard
sudo systemctl stop dashboard

# Ver logs recientes
sudo journalctl -u dashboard -n 50
```

---

## 🐛 Troubleshooting

### Dashboard no arranca

**Problema:** `systemctl status dashboard` muestra error

**Soluciones:**

1. **Verificar dependencias**
   ```bash
   cd ~/CriptoBot
   source .venv/bin/activate
   python -c "import flask, plotly"
   ```

2. **Verificar permisos**
   ```bash
   ls -la dashboard.py bot_state.json trades_production.csv
   ```

3. **Ver logs detallados**
   ```bash
   sudo journalctl -u dashboard -n 100 --no-pager
   ```

### No se conecta al dashboard

**Problema:** Browser no carga `http://IP:5000`

**Soluciones:**

1. **Verificar que el dashboard esté corriendo**
   ```bash
   sudo systemctl status dashboard
   # Debe mostrar: Active: active (running)
   ```

2. **Verificar que Flask escucha en 0.0.0.0**
   ```bash
   sudo netstat -tulpn | grep 5000
   # Debe mostrar: 0.0.0.0:5000
   ```

3. **Verificar firewall**
   ```bash
   sudo ufw status
   # Si está activo, añadir regla:
   sudo ufw allow 5000
   ```

4. **Probar localmente primero**
   ```bash
   curl http://localhost:5000
   # Debe devolver HTML del dashboard
   ```

### Gráficos no cargan

**Problema:** Dashboard carga pero gráficos no aparecen

**Soluciones:**

1. **Verificar API endpoints**
   ```bash
   curl http://localhost:5000/api/status
   curl http://localhost:5000/api/chart/ETH
   ```

2. **Ver consola del navegador** (F12 → Console)
   - Buscar errores de JavaScript
   - Verificar que Plotly cargó correctamente

3. **Verificar datos existen**
   ```bash
   cat bot_state.json
   cat trades_production.csv
   ```

### Datos no actualizan

**Problema:** Dashboard muestra datos antiguos

**Causas posibles:**
- Bot detenido
- Archivos no se están actualizando

**Soluciones:**

1. **Verificar que el bot esté corriendo**
   ```bash
   sudo systemctl status bot
   ```

2. **Verificar última modificación de archivos**
   ```bash
   ls -lt bot_state.json trades_production.csv
   ```

3. **Force refresh en el navegador**
   - Chrome/Firefox: `Ctrl + Shift + R`
   - Safari: `Cmd + Shift + R`

---

## 🔒 Seguridad

> **⚠️ IMPORTANTE**: El dashboard está configurado para escuchar en `0.0.0.0:5000`, lo que significa que es accesible desde cualquier IP.

### Recomendaciones de Seguridad

1. **Usar con VPN** o **SSH Tunnel**:
   ```bash
   # En tu PC local
   ssh -L 5000:localhost:5000 j0s3m4@IP_VPS
   # Luego accede a http://localhost:5000
   ```

2. **Firewall restrictivo** (solo tu IP):
   ```bash
   sudo ufw allow from TU_IP_LOCAL to any port 5000
   ```

3. **Nginx con autenticación básica** (avanzado):
   - Configurar Nginx como proxy reverso
   - Añadir HTTP Basic Auth
   - Opcional: HTTPS con Let's Encrypt

---

## 📁 Estructura de Archivos

```
CriptoBot/
├── dashboard.py          # Flask backend
├── dashboard.service     # systemd service
├── templates/
│   └── dashboard.html    # HTML template
└── static/
    ├── dashboard.js      # JavaScript (charts, updates)
    └── style.css         # CSS (dark theme)
```

---

## 🎨 Personalización

### Cambiar Intervalo de Actualización

Edita `static/dashboard.js`:

```javascript
const REFRESH_INTERVAL = 30000; // Cambiar a 60000 para 1 minuto
```

### Cambiar Puerto

Edita `dashboard.py`:

```python
app.run(host='0.0.0.0', port=5000, debug=False)  # Cambiar 5000
```

No olvides actualizar el servicio:
```bash
sudo nano /etc/systemd/system/dashboard.service
# Cambiar ExecStart si es necesario
sudo systemctl daemon-reload
sudo systemctl restart dashboard
```

### Modificar Colores

Edita `static/style.css` para cambiar el tema.

---

## 💡 Tips

1. **Usa pantalla completa** (F11) para mejor visualización
2. **Abre en pestaña dedicada** y déjala visible en segundo monitor
3. **Bookmarkea la URL** para acceso rápido
4. **Zoom del gráfico**: Útil para analizar períodos específicos
5. **Double-click en gráfico**: Resetea zoom

---

## 🔗 Enlaces Relacionados

- [README.md](README.md) - Guía principal del bot
- [INSTALL.md](INSTALL.md) - Instalación del bot
- [TELEGRAM.md](TELEGRAM.md) - Notificaciones Telegram
- [UPDATE.md](UPDATE.md) - Sistema de actualizaciones

---

**🎉 ¡Disfruta monitoreando tu bot con el dashboard!**
