#!/bin/bash
# Script wrapper para el bot que muestra la versión al iniciar

# Leer versión del archivo VERSION
VERSION=$(cat /home/j0s3m4/CriptoBot/VERSION | tr -d '\n\r')

# ============================================================================
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
echo "Trading Bot v$VERSION"
echo "========================================"

# Ejecutar el bot
exec /home/j0s3m4/CriptoBot/.venv/bin/python /home/j0s3m4/CriptoBot/bot_production.py
