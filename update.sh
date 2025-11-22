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
# 5. VERIFICAR SI EL BOT ESTÁ CORRIENDO
# ============================================================================
BOT_RUNNING=false
if systemctl is-active --quiet bot.service 2>/dev/null; then
    BOT_RUNNING=true
    print_warning "El bot está corriendo como servicio"
    echo "Se detendrá temporalmente para aplicar la actualización"
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
# 7. DETENER EL BOT SI ESTÁ CORRIENDO
# ============================================================================
if [ "$BOT_RUNNING" = true ]; then
    print_info "Deteniendo el bot..."
    sudo systemctl stop bot.service
    print_success "Bot detenido"
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

# ============================================================================
# 9. APLICAR ACTUALIZACIÓN
# ============================================================================
print_header "APLICANDO ACTUALIZACIÓN"

print_info "Descargando cambios desde GitHub..."
git pull origin main

print_success "Código actualizado"

# ============================================================================
# 10. RESTAURAR ARCHIVOS DE CONFIGURACIÓN
# ============================================================================
print_info "Restaurando configuración..."

if [ -f ".env.tmp" ]; then
    mv .env.tmp .env
    print_success "Archivo .env restaurado"
fi

if [ -f "bot_state.json.tmp" ]; then
    mv bot_state.json.tmp bot_state.json
    print_success "Archivo bot_state.json restaurado"
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
# 12. REINICIAR EL BOT
# ============================================================================
if [ "$BOT_RUNNING" = true ]; then
    print_header "REINICIANDO BOT"
    
    print_info "Iniciando el bot..."
    sudo systemctl start bot.service
    
    # Esperar un momento
    sleep 3
    
    # Verificar que arrancó correctamente
    if systemctl is-active --quiet bot.service; then
        print_success "Bot reiniciado exitosamente"
    else
        print_error "El bot no pudo iniciar correctamente"
        echo ""
        echo "Verifica el estado con:"
        echo "  sudo systemctl status bot"
        echo "  sudo journalctl -u bot -n 50"
        echo ""
        echo "Si hay problemas, puedes restaurar el backup:"
        echo "  sudo systemctl stop bot"
        echo "  rm -rf $SCRIPT_DIR/*"
        echo "  cp -r $BACKUP_DIR/* $SCRIPT_DIR/"
        echo "  sudo systemctl start bot"
        exit 1
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

if [ "$BOT_RUNNING" = true ]; then
    echo -e "\n${BLUE}Estado del bot:${NC}"
    sudo systemctl status bot --no-pager -l | head -10
fi

echo -e "\n${BLUE}Comandos útiles:${NC}"
echo "  sudo systemctl status bot    - Ver estado"
echo "  sudo journalctl -u bot -f    - Ver logs en tiempo real"
echo "  cat bot_state.json           - Ver estado actual"

echo -e "\n${GREEN}¡Actualización exitosa!${NC}\n"
