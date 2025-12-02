#!/bin/bash

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "🚀 Iniciando Bots Neuronales..."

# 1. Bot BTC (Especialista BTC+ETH)
echo "   - Iniciando Bot BTC..."
nohup python bot_neural.py --model BTC_4h --symbols BTC/USDT --id BTC > log_btc.txt 2>&1 &
echo $! > pid_btc.txt

# 2. Bot ETH (Generalista)
echo "   - Iniciando Bot ETH..."
nohup python bot_neural.py --model GENERAL_4h_v2 --symbols ETH/USDT --id ETH > log_eth.txt 2>&1 &
echo $! > pid_eth.txt

# 3. Bot SOL (Grupo Volátil)
echo "   - Iniciando Bot SOL..."
nohup python bot_neural.py --model SOL_GROUP_4h --symbols SOL/USDT --id SOL > log_sol.txt 2>&1 &
echo $! > pid_sol.txt

# 4. Telegram Bot (Manejador)
echo "   - Iniciando Telegram Bot..."
nohup python telegram_bot_handler.py > log_telegram.txt 2>&1 &
echo $! > pid_telegram.txt

echo "✅ Todos los servicios iniciados en segundo plano."
echo "   Logs disponibles en: log_btc.txt, log_eth.txt, log_sol.txt, log_telegram.txt"
