#!/bin/bash

echo "🛑 Deteniendo servicios..."

# Función para matar proceso por archivo PID
kill_process() {
    if [ -f "$1" ]; then
        pid=$(cat "$1")
        if ps -p $pid > /dev/null; then
            kill $pid
            echo "   - Proceso $pid detenido ($1)"
        else
            echo "   - Proceso $pid no encontrado ($1)"
        fi
        rm "$1"
    fi
}

kill_process "pid_btc.txt"
kill_process "pid_eth.txt"
kill_process "pid_sol.txt"
kill_process "pid_telegram.txt"

# Limpieza adicional por si acaso (opcional, cuidado si hay otros python corriendo)
# pkill -f "bot_neural.py"
# pkill -f "telegram_bot_handler.py"

echo "✅ Servicios detenidos."
