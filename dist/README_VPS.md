# 🚀 Guía de Despliegue en VPS

Esta carpeta `dist/` contiene todo lo necesario para ejecutar tus bots neuronales en un servidor.

## 📋 Pasos Previos (En tu PC Local)

1.  **Comprimir:** Comprime el contenido de esta carpeta `dist/` en un archivo `.zip` o `.tar.gz`.
    *   Ejemplo: `zip -r bot_deploy.zip dist/`

2.  **Subir al VPS:** Usa SCP o FileZilla para subir el archivo al servidor.
    *   `scp bot_deploy.zip usuario@tu_vps:/home/usuario/`

## 🛠️ Instalación (En el VPS)

1.  **Descomprimir:**
    ```bash
    unzip bot_deploy.zip
    cd dist
    ```

2.  **Dar permisos de ejecución:**
    ```bash
    chmod +x install.sh start_bots.sh stop_bots.sh
    ```

3.  **Instalar:**
    Este script creará el entorno virtual e instalará las dependencias.
    ```bash
    ./install.sh
    ```

4.  **Configurar Variables:**
    Renombra el ejemplo y edita tus claves API y Token de Telegram.
    ```bash
    cp .env.example .env
    nano .env
    ```

## ▶️ Ejecución

### Iniciar Bots
Ejecuta el script de inicio. Esto lanzará 4 procesos en segundo plano (BTC, ETH, SOL y Telegram).
```bash
./start_bots.sh
```

### Verificar Estado
Puedes ver si los procesos están corriendo con:
```bash
ps aux | grep python
```
O revisando los logs generados:
```bash
tail -f log_telegram.txt
tail -f log_btc.txt
```

### Detener Bots
Para parar todo ordenadamente:
```bash
./stop_bots.sh
```

## 🤖 Comandos de Telegram

Una vez iniciados, ve a tu bot de Telegram y usa:
- `/start`: Menú principal.
- `/status`: Ver estado de los 3 bots.
- `/posiciones`: Ver operaciones abiertas.
