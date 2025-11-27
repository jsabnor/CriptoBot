#!/bin/bash

# ============================================================================
# SCRIPT DE ACTUALIZACIÓN AUTOMATIZADA - BOT DE TRADING
# ============================================================================
# Actualiza el bot desde GitHub con backup automático y reinicio del servicio
# Uso: ./update.sh
# ============================================================================

set -e  # Salir si hay algún error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Funciones
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

# Obtener directorio del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_header "ACTUALIZACIÓN DEL BOT DE TRADING"

# ============================================================================
# 1. VERIFICAR QUE ES UN REPOSITORIO GIT
# ============================================================================
print_info "Verificando configuración de git..."

if [ ! -d ".git" ]; then
    print_warning "Este directorio no es un repositorio git"
    echo -e "\n¿Deseas convertirlo en un repositorio git conectado a GitHub?"
    read -p "Esto permitirá actualizaciones automáticas (s/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        print_info "Inicializando repositorio git..."
        git init
        git remote add origin https://github.com/jsabnor/CriptoBot.git
        git fetch origin
        git reset --hard origin/main
        print_success "Repositorio git configurado"
    else
        print_error "Cancelado por el usuario"
        exit 1
    fi
fi

# Verificar remote
if ! git remote get-url origin >/dev/null 2>&1; then
    print_warning "No hay remote configurado"
    print_info "Configurando remote..."
    git remote add origin https://github.com/jsabnor/CriptoBot.git
    print_success "Remote configurado"
fi

# ============================================================================
# 2. OBTENER ACTUALIZACIONES DEL REMOTO
# ============================================================================
print_info "Obteniendo información de GitHub..."
git fetch origin

# Comparar versiones
LOCAL=$(git rev-parse @ 2>/dev/null)
REMOTE=$(git rev-parse @{u} 2>/dev/null)

if [ "$LOCAL" = "$REMOTE" ]; then
    print_success "Tu bot ya está actualizado"
    LOCAL_VERSION=$(cat VERSION 2>/dev/null || echo "desconocida")
    echo -e "Versión actual: ${GREEN}v$LOCAL_VERSION${NC}"
    exit 0
fi

# ============================================================================
# 3. MOSTRAR CAMBIOS DISPONIBLES
# ============================================================================
print_header "ACTUALIZACIONES DISPONIBLES"

# Versiones
LOCAL_VERSION=$(cat VERSION 2>/dev/null || echo "desconocida")
REMOTE_VERSION=$(git show origin/main:VERSION 2>/dev/null || echo "desconocida")

echo -e "Versión actual:  ${YELLOW}v$LOCAL_VERSION${NC}"
echo -e "Versión nueva:   ${GREEN}v$REMOTE_VERSION${NC}"

echo -e "\n${BLUE}Commits nuevos:${NC}"
git log --oneline --graph --color=always HEAD..origin/main | head -10

echo -e "\n${BLUE}Archivos que cambiarán:${NC}"
git diff --name-status HEAD origin/main | head -20

# ============================================================================
# 4. SOLICITAR CONFIRMACIÓN
# ============================================================================
echo -e "\n${YELLOW}¿Deseas aplicar esta actualización?${NC}"
echo "Se creará un backup automático antes de actualizar"
read -p "Continuar (s/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    print_info "Actualización cancelada por el usuario"
    exit 0
fi

# ============================================================================
# 5. VERIFICAR SI LOS BOTS ESTÁN CORRIENDO
# ============================================================================
BOT_ADX_RUNNING=false
BOT_EMA_RUNNING=false
BOT_TELEGRAM_RUNNING=false

if systemctl is-active --quiet bot.service 2>/dev/null; then
    BOT_ADX_RUNNING=true
    print_warning "El bot ADX está corriendo como servicio"
fi

if systemctl is-active --quiet bot_ema.service 2>/dev/null; then
    BOT_EMA_RUNNING=true
    print_warning "El bot EMA está corriendo como servicio"
fi

if systemctl is-active --quiet telegram_bot.service 2>/dev/null; then
    BOT_TELEGRAM_RUNNING=true
    print_warning "El bot de Telegram está corriendo como servicio"
fi

if [ "$BOT_ADX_RUNNING" = true ] || [ "$BOT_EMA_RUNNING" = true ] || [ "$BOT_TELEGRAM_RUNNING" = true ]; then
    echo "Los bots se detendrán temporalmente para aplicar la actualización"
fi

# ============================================================================
# 6. CREAR BACKUP
# ============================================================================
print_header "CREANDO BACKUP"

BACKUP_DIR="${SCRIPT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
print_info "Creando backup en: $BACKUP_DIR"

# Crear directorio de backup
mkdir -p "$BACKUP_DIR"

# Copiar archivos importantes (excluir .venv, .git)
rsync -a \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/' \
    "$SCRIPT_DIR/" "$BACKUP_DIR/"

print_success "Backup creado exitosamente"
print_info "Ubicación: $BACKUP_DIR"

# ============================================================================
# 7. DETENER LOS BOTS SI ESTÁN CORRIENDO  
# ============================================================================
if [ "$BOT_ADX_RUNNING" = true ]; then
    print_info "Deteniendo el bot ADX..."
    sudo systemctl stop bot.service
    print_success "Bot ADX detenido"
fi

if [ "$BOT_EMA_RUNNING" = true ]; then
    print_info "Deteniendo el bot EMA..."
    sudo systemctl stop bot_ema.service
    print_success "Bot EMA detenido"
fi

if [ "$BOT_TELEGRAM_RUNNING" = true ]; then
    print_info "Deteniendo el bot de Telegram..."
    sudo systemctl stop telegram_bot.service
    print_success "Bot de Telegram detenido"
fi

# ============================================================================
# 8. PRESERVAR ARCHIVOS DE CONFIGURACIÓN
# ============================================================================
print_info "Preservando archivos de configuración..."

# Backup temporal de archivos que no deben sobrescribirse
if [ -f ".env" ]; then
    cp .env .env.tmp
fi
if [ -f "bot_state.json" ]; then
    cp bot_state.json bot_state.json.tmp
fi
if [ -f "bot_state_ema.json" ]; then
    cp bot_state_ema.json bot_state_ema.json.tmp
fi

# ============================================================================
# 9. APLICAR ACTUALIZACIÓN
# ============================================================================
print_header "APLICANDO ACTUALIZACIÓN"

# Guardar hash del update.sh actual para detectar si cambió
CURRENT_UPDATE_HASH=$(md5sum "$0" 2>/dev/null | cut -d' ' -f1)

# Guardar cambios locales temporalmente (especialmente update.sh)
print_info "Guardando cambios locales temporalmente..."
git stash push -m "Auto-stash before update $(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

print_info "Descargando cambios desde GitHub..."
if ! git pull origin main; then
    print_error "Error al descargar cambios"
    print_warning "Intentando recuperar cambios guardados..."
    git stash pop 2>/dev/null || true
    exit 1
fi

# Restaurar cambios locales si los había
if git stash list | grep -q "Auto-stash before update"; then
    print_info "Restaurando cambios locales..."
    git stash pop 2>/dev/null || true
fi

print_success "Código actualizado"

# ============================================================================
# 9.5. VERIFICAR SI UPDATE.SH CAMBIÓ Y RE-EJECUTAR
# ============================================================================
NEW_UPDATE_HASH=$(md5sum "$0" 2>/dev/null | cut -d' ' -f1)

if [ "$CURRENT_UPDATE_HASH" != "$NEW_UPDATE_HASH" ]; then
    print_warning "El script update.sh ha sido actualizado"
    print_info "Re-ejecutando con la nueva versión..."
    
    # Marcar que ya estamos en una re-ejecución para evitar bucles infinitos
    export UPDATE_REEXECUTED="true"
    
    # Re-ejecutar el script actualizado
    exec bash "$0" "$@"
fi

# Si llegamos aquí y ya fue re-ejecutado, continuar normalmente
if [ "$UPDATE_REEXECUTED" = "true" ]; then
    print_info "Ejecutando versión actualizada del script"
    unset UPDATE_REEXECUTED
fi

# ============================================================================
# 10. RESTAURAR ARCHIVOS DE CONFIGURACIÓN
# ============================================================================
print_info "Restaurando configuración..."

# Obtener usuario actual (el que ejecutó sudo)
ACTUAL_USER=${SUDO_USER:-$USER}

if [ -f ".env.tmp" ]; then
    mv .env.tmp .env
    # Restaurar permisos y propietario correctos
    chmod 600 .env
    chown $ACTUAL_USER:$ACTUAL_USER .env
    print_success "Archivo .env restaurado"
fi

if [ -f "bot_state.json.tmp" ]; then
    mv bot_state.json.tmp bot_state.json
    chown $ACTUAL_USER:$ACTUAL_USER bot_state.json
    print_success "Archivo bot_state.json restaurado"
fi

if [ -f "bot_state_ema.json.tmp" ]; then
    mv bot_state_ema.json.tmp bot_state_ema.json
    chown $ACTUAL_USER:$ACTUAL_USER bot_state_ema.json
    print_success "Archivo bot_state_ema.json restaurado"
fi

# ============================================================================
# 11. ACTUALIZAR DEPENDENCIAS SI ES NECESARIO
# ============================================================================
print_header "VERIFICANDO DEPENDENCIAS"

# Verificar si requirements.txt cambió
if git diff --name-only HEAD@{1} HEAD | grep -q "requirements.txt"; then
    print_warning "requirements.txt ha cambiado"
    print_info "Actualizando dependencias Python..."
    
    source .venv/bin/activate
    pip install -r requirements.txt --upgrade --quiet
    
    print_success "Dependencias actualizadas"
else
    print_info "No hay cambios en las dependencias"
fi

# ============================================================================
# 11.5. ACTUALIZAR SERVICIOS SYSTEMD
# ============================================================================
print_header "ACTUALIZANDO SERVICIOS SYSTEMD"

# Copiar archivos de servicio si existen
if [ -f "bot.service" ]; then
    print_info "Actualizando bot.service..."
    sudo cp bot.service /etc/systemd/system/bot.service
    print_success "bot.service actualizado"
fi

if [ -f "bot_ema.service" ]; then
    print_info "Actualizando bot_ema.service..."
    sudo cp bot_ema.service /etc/systemd/system/bot_ema.service
    print_success "bot_ema.service actualizado"
fi

if [ -f "telegram_bot.service" ]; then
    print_info "Actualizando telegram_bot.service..."
    sudo cp telegram_bot.service /etc/systemd/system/telegram_bot.service
    print_success "telegram_bot.service actualizado"
fi

# Recargar systemd
print_info "Recargando systemd..."
sudo systemctl daemon-reload
print_success "Systemd recargado"

# ============================================================================
# 11.6. CORREGIR PERMISOS DE TODOS LOS ARCHIVOS
# ============================================================================
print_info "Corrigiendo permisos de archivos..."

# Obtener usuario actual (el que ejecutó sudo)
ACTUAL_USER=${SUDO_USER:-$USER}

# Cambiar propietario de todo el directorio al usuario correcto
chown -R $ACTUAL_USER:$ACTUAL_USER "$SCRIPT_DIR"

# Asegurar permisos específicos para archivos sensibles
if [ -f ".env" ]; then
    chmod 600 .env
fi

# Hacer scripts ejecutables
chmod +x check_updates.sh 2>/dev/null || true
chmod +x update.sh 2>/dev/null || true
chmod +x install.sh 2>/dev/null || true
chmod +x start_bot.sh 2>/dev/null || true

print_success "Permisos corregidos"

# ============================================================================
# 12. REINICIAR LOS BOTS
# ============================================================================
if [ "$BOT_ADX_RUNNING" = true ] || [ "$BOT_EMA_RUNNING" = true ] || [ "$BOT_TELEGRAM_RUNNING" = true ]; then
    print_header "REINICIANDO BOTS"
    
    if [ "$BOT_ADX_RUNNING" = true ]; then
        print_info "Iniciando el bot ADX..."
        sudo systemctl start bot.service
        sleep 2
        
        if systemctl is-active --quiet bot.service; then
            print_success "Bot ADX reiniciado exitosamente"
        else
            print_error "El bot ADX no pudo iniciar correctamente"
            echo "Verifica: sudo systemctl status bot"
        fi
    fi
    
    if [ "$BOT_EMA_RUNNING" = true ]; then
        print_info "Iniciando el bot EMA..."
        sudo systemctl start bot_ema.service
        sleep 2
        
        if systemctl is-active --quiet bot_ema.service; then
            print_success "Bot EMA reiniciado exitosamente"
        else
            print_error "El bot EMA no pudo iniciar correctamente"
            echo "Verifica: sudo systemctl status bot_ema"
        fi
    fi
    
    if [ "$BOT_TELEGRAM_RUNNING" = true ]; then
        print_info "Iniciando el bot de Telegram..."
        sudo systemctl start telegram_bot.service
        sleep 2
        
        if systemctl is-active --quiet telegram_bot.service; then
            print_success "Bot de Telegram reiniciado exitosamente"
        else
            print_error "El bot de Telegram no pudo iniciar correctamente"
            echo "Verifica: sudo systemctl status telegram_bot"
        fi
    fi
fi

# ============================================================================
# 13. RESUMEN
# ============================================================================
print_header "ACTUALIZACIÓN COMPLETADA"

NEW_VERSION=$(cat VERSION)
print_success "Bot actualizado a versión v$NEW_VERSION"

echo -e "\n${GREEN}Información:${NC}"
echo -e "  Versión anterior: v$LOCAL_VERSION"
echo -e "  Versión actual:   v$NEW_VERSION"
echo -e "  Backup guardado:  $BACKUP_DIR"

if [ "$BOT_ADX_RUNNING" = true ] || [ "$BOT_EMA_RUNNING" = true ] || [ "$BOT_TELEGRAM_RUNNING" = true ]; then
    echo -e "\n${BLUE}Estado de los bots:${NC}"
    
    if [ "$BOT_ADX_RUNNING" = true ]; then
        echo -e "\n${GREEN}Bot ADX:${NC}"
        sudo systemctl status bot --no-pager -l | head -10
    fi
    
    if [ "$BOT_EMA_RUNNING" = true ]; then
        echo -e "\n${GREEN}Bot EMA:${NC}"
        sudo systemctl status bot_ema --no-pager -l | head -10
    fi
    
    if [ "$BOT_TELEGRAM_RUNNING" = true ]; then
        echo -e "\n${GREEN}Bot Telegram:${NC}"
        sudo systemctl status telegram_bot --no-pager -l | head -10
    fi
fi

echo -e "\n${BLUE}Comandos útiles:${NC}"
echo "  sudo systemctl status bot bot_ema telegram_bot  - Ver estado de los bots"
echo "  sudo journalctl -u bot -f                       - Ver logs bot ADX"
echo "  sudo journalctl -u bot_ema -f                   - Ver logs bot EMA"
echo "  sudo journalctl -u telegram_bot -f              - Ver logs bot Telegram"
echo "  cat bot_state.json                              - Ver estado bot ADX"
echo "  cat bot_state_ema.json                          - Ver estado bot EMA"

echo -e "\n${GREEN}¡Actualización exitosa!${NC}\n"
