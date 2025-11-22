#!/bin/bash

# ============================================================================
# SCRIPT DE INSTALACIÓN AUTOMATIZADA - BOT DE TRADING v1.0
# ============================================================================
# Este script instala todo lo necesario para ejecutar el bot en un VPS Ubuntu/Debian
# Uso: sudo ./install.sh
# ============================================================================

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones de utilidad
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# 1. VERIFICACIÓN DE SISTEMA
# ============================================================================
print_header "1/10 - VERIFICACIÓN DE SISTEMA"

# Verificar que se ejecuta con sudo
if [ "$EUID" -ne 0 ]; then 
    print_error "Este script debe ejecutarse con sudo"
    echo "Uso: sudo ./install.sh"
    exit 1
fi

# Detectar sistema operativo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VERSION=$VERSION_ID
    print_success "Sistema detectado: $OS $VERSION"
else
    print_error "No se pudo detectar el sistema operativo"
    exit 1
fi

# Verificar que es Ubuntu o Debian
if [[ ! "$OS" =~ (Ubuntu|Debian) ]]; then
    print_warning "Este script está diseñado para Ubuntu/Debian"
    read -p "¿Deseas continuar de todas formas? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# 2. ACTUALIZAR SISTEMA
# ============================================================================
print_header "2/10 - ACTUALIZANDO SISTEMA"

apt update
apt upgrade -y
print_success "Sistema actualizado"

# ============================================================================
# 3. INSTALAR PYTHON Y HERRAMIENTAS
# ============================================================================
print_header "3/10 - INSTALANDO PYTHON 3.9+"

# Verificar si Python 3.9+ ya está instalado
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )(.+)')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
    print_success "Python $PYTHON_VERSION ya instalado"
else
    print_info "Instalando Python 3.9+"
    apt install -y python3 python3-pip python3-venv
    print_success "Python instalado"
fi

# Verificar pip
if command -v pip3 &> /dev/null; then
    print_success "pip3 instalado"
else
    apt install -y python3-pip
    print_success "pip3 instalado"
fi

# ============================================================================
# 4. DETERMINAR DIRECTORIO DE INSTALACIÓN
# ============================================================================
print_header "4/10 - CONFIGURANDO DIRECTORIO"

# Obtener el directorio desde donde se ejecuta el script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BOT_DIR="$SCRIPT_DIR"

print_info "Directorio del bot: $BOT_DIR"

# Verificar que bot_production.py existe
if [ ! -f "$BOT_DIR/bot_production.py" ]; then
    print_error "No se encontró bot_production.py en $BOT_DIR"
    exit 1
fi

print_success "Archivos del bot encontrados"

# ============================================================================
# 5. CREAR ENTORNO VIRTUAL
# ============================================================================
print_header "5/10 - CREANDO ENTORNO VIRTUAL"

cd "$BOT_DIR"

if [ -d ".venv" ]; then
    print_warning "Entorno virtual ya existe, se recreará"
    rm -rf .venv
fi

python3 -m venv .venv
print_success "Entorno virtual creado"

# ============================================================================
# 6. INSTALAR DEPENDENCIAS PYTHON
# ============================================================================
print_header "6/10 - INSTALANDO DEPENDENCIAS PYTHON"

source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Verificar si existe requirements.txt
if [ -f "requirements.txt" ]; then
    print_info "Instalando desde requirements.txt..."
    pip install -r requirements.txt
else
    print_info "Instalando dependencias manualmente..."
    pip install pandas>=1.5.0 numpy>=1.23.0 ccxt>=4.0.0 python-dotenv>=1.0.0
fi

print_success "Dependencias instaladas"

# Mostrar paquetes instalados
print_info "Paquetes instalados:"
pip list | grep -E "pandas|numpy|ccxt|python-dotenv"

# ============================================================================
# 7. CONFIGURAR VARIABLES DE ENTORNO
# ============================================================================
print_header "7/10 - CONFIGURANDO VARIABLES DE ENTORNO"

if [ -f ".env" ]; then
    print_warning "El archivo .env ya existe"
    read -p "¿Deseas reconfigurarlo? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        CONFIGURE_ENV=true
    else
        CONFIGURE_ENV=false
    fi
else
    CONFIGURE_ENV=true
fi

if [ "$CONFIGURE_ENV" = true ]; then
    echo -e "\n${YELLOW}Necesitas las claves API de Binance.${NC}"
    echo -e "${YELLOW}Obtén tus claves en: https://www.binance.com/en/my/settings/api-management${NC}\n"
    
    read -p "Ingresa tu BINANCE API KEY: " API_KEY
    read -p "Ingresa tu BINANCE API SECRET: " API_SECRET
    
    echo -e "\n${YELLOW}Selecciona el modo de trading:${NC}"
    echo "1) paper - Simulación (NO usa dinero real) - RECOMENDADO"
    echo "2) live - Trading real (usa dinero real)"
    read -p "Elige (1/2): " -n 1 -r MODE_CHOICE
    echo
    
    if [ "$MODE_CHOICE" = "2" ]; then
        TRADING_MODE="live"
        print_warning "ATENCIÓN: Modo LIVE seleccionado - usará dinero real"
    else
        TRADING_MODE="paper"
        print_success "Modo PAPER seleccionado - simulación segura"
    fi
    
    read -p "Capital por par en EUR (default: 50): " CAPITAL
    CAPITAL=${CAPITAL:-50.0}
    
    # Crear archivo .env
    cat > .env << EOF
# Configuración del Bot de Trading v1.0
# Generado automáticamente: $(date)

# Claves API de Binance
BINANCE_API_KEY=$API_KEY
BINANCE_API_SECRET=$API_SECRET

# Modo de trading
TRADING_MODE=$TRADING_MODE

# Capital inicial por par (EUR)
CAPITAL_PER_PAIR=$CAPITAL
EOF

    chmod 600 .env
    print_success "Archivo .env creado y protegido"
else
    print_info "Usando archivo .env existente"
fi

# ============================================================================
# 8. TEST DE CONECTIVIDAD
# ============================================================================
print_header "8/10 - TEST DE CONECTIVIDAD"

print_info "Probando conexión con Binance API..."

# Crear script de test temporal
cat > /tmp/test_binance.py << 'EOF'
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("ERROR: Claves API no configuradas en .env")
    exit(1)

try:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })
    
    # Test de conectividad
    exchange.load_markets()
    balance = exchange.fetch_balance()
    
    print("✓ Conexión exitosa con Binance")
    print(f"✓ Cuenta: {balance['info']['accountType']}")
    exit(0)
except Exception as e:
    print(f"✗ Error de conexión: {e}")
    exit(1)
EOF

if python /tmp/test_binance.py; then
    print_success "Conexión con Binance exitosa"
else
    print_error "No se pudo conectar con Binance"
    print_warning "Verifica tus claves API en el archivo .env"
fi

rm /tmp/test_binance.py

# ============================================================================
# 9. CONFIGURAR SERVICIO SYSTEMD (OPCIONAL)
# ============================================================================
print_header "9/10 - CONFIGURAR SERVICIO SYSTEMD"

echo -e "${YELLOW}¿Deseas configurar el bot como servicio systemd?${NC}"
echo "Esto permitirá:"
echo "  - Auto-inicio al arrancar el sistema"
echo "  - Reinicio automático si el bot falla"
echo "  - Logs centralizados en journalctl"
read -p "Configurar servicio? (s/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Ss]$ ]]; then
    # Obtener usuario actual (el que ejecutó sudo)
    ACTUAL_USER=${SUDO_USER:-$USER}
    
    # Crear archivo de servicio
    cat > /etc/systemd/system/bot.service << EOF
[Unit]
Description=Trading Bot v1.0
After=network.target

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$BOT_DIR
Environment="PATH=$BOT_DIR/.venv/bin"
ExecStart=$BOT_DIR/.venv/bin/python $BOT_DIR/bot_production.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Recargar systemd
    systemctl daemon-reload
    systemctl enable bot.service
    
    print_success "Servicio systemd configurado"
    print_info "Comandos útiles:"
    echo "  sudo systemctl start bot    - Iniciar bot"
    echo "  sudo systemctl stop bot     - Detener bot"
    echo "  sudo systemctl status bot   - Ver estado"
    echo "  sudo journalctl -u bot -f   - Ver logs"
    
    read -p "¿Deseas iniciar el bot ahora? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        systemctl start bot
        sleep 2
        systemctl status bot --no-pager
    fi
else
    print_info "Servicio systemd omitido"
fi

# ============================================================================
# 10. RESUMEN Y PRÓXIMOS PASOS
# ============================================================================
print_header "10/10 - INSTALACIÓN COMPLETADA"

print_success "¡Bot de Trading v1.0 instalado exitosamente!"

echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}INFORMACIÓN DE LA INSTALACIÓN${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "📁 Directorio: $BOT_DIR"
echo -e "🐍 Python: $(python3 --version)"
echo -e "🔧 Modo: $TRADING_MODE"
echo -e "💰 Capital por par: ${CAPITAL} EUR"
echo -e "💸 Capital total: $((${CAPITAL%.*} * 4)) EUR (4 pares)"

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}PRÓXIMOS PASOS${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

if systemctl is-enabled bot.service &> /dev/null; then
    echo -e "✅ El bot está configurado como servicio"
    echo -e "\n📊 Monitorear el bot:"
    echo -e "   sudo systemctl status bot"
    echo -e "   sudo journalctl -u bot -f"
    echo -e "\n🔄 Gestionar el bot:"
    echo -e "   sudo systemctl start bot"
    echo -e "   sudo systemctl stop bot"
    echo -e "   sudo systemctl restart bot"
else
    echo -e "🚀 Ejecutar el bot manualmente:"
    echo -e "   cd $BOT_DIR"
    echo -e "   source .venv/bin/activate"
    echo -e "   python bot_production.py"
fi

echo -e "\n📂 Archivos importantes:"
echo -e "   $BOT_DIR/bot_state.json         - Estado actual"
echo -e "   $BOT_DIR/trades_production.csv  - Historial trades"
echo -e "   $BOT_DIR/.env                   - Configuración (NO compartir)"

echo -e "\n⚠️  ${YELLOW}RECOMENDACIONES:${NC}"
if [ "$TRADING_MODE" = "paper" ]; then
    echo -e "   ✓ Estás en modo PAPER (simulación segura)"
    echo -e "   ✓ Monitorea 1-2 meses antes de pasar a LIVE"
else
    echo -e "   ${RED}⚠ Estás en modo LIVE - usas dinero real${NC}"
    echo -e "   ⚠ Solo invierte lo que puedas perder"
fi
echo -e "   ✓ Revisa bot_state.json diariamente"
echo -e "   ✓ Haz backup de los archivos regularmente"
echo -e "   ✓ Nunca compartas tus claves API"

echo -e "\n📚 Documentación completa: $BOT_DIR/INSTALL.md"

echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}¡Instalación completada exitosamente!${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}\n"
