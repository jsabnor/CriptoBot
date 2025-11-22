#!/bin/bash

# ============================================================================
# SCRIPT DE VERIFICACIÓN DE ACTUALIZACIONES - BOT DE TRADING
# ============================================================================
# Verifica si hay actualizaciones disponibles en GitHub sin aplicarlas
# Uso: ./check_updates.sh
# ============================================================================

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Obtener directorio del script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}Verificando actualizaciones...${NC}\n"

# Verificar si es un repositorio git
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠ Este directorio no es un repositorio git${NC}"
    echo "Para habilitar actualizaciones automáticas, ejecuta:"
    echo "  git init"
    echo "  git remote add origin https://github.com/jsabnor/CriptoBot.git"
    echo "  git fetch origin"
    exit 1
fi

# Obtener información del repositorio remoto
git fetch origin --quiet 2>/dev/null

# Verificar si hay remote configurado
if ! git remote get-url origin >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ No hay remote configurado${NC}"
    echo "Configura el remote con:"
    echo "  git remote add origin https://github.com/jsabnor/CriptoBot.git"
    exit 1
fi

# Comparar commits local vs remoto
LOCAL=$(git rev-parse @ 2>/dev/null)
REMOTE=$(git rev-parse @{u} 2>/dev/null)

# Leer versión local
if [ -f "VERSION" ]; then
    LOCAL_VERSION=$(cat VERSION)
else
    LOCAL_VERSION="desconocida"
fi

if [ "$LOCAL" = "$REMOTE" ]; then
    echo -e "${GREEN}✓ Tu bot está actualizado${NC}"
    echo -e "Versión actual: ${GREEN}v$LOCAL_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Hay actualizaciones disponibles${NC}"
    echo -e "Versión local: ${YELLOW}v$LOCAL_VERSION${NC}"
    
    # Obtener versión remota
    REMOTE_VERSION=$(git show origin/main:VERSION 2>/dev/null || echo "desconocida")
    echo -e "Versión remota: ${GREEN}v$REMOTE_VERSION${NC}"
    
    echo ""
    echo -e "${BLUE}Cambios disponibles:${NC}"
    git log --oneline HEAD..origin/main --color=always | head -5
    
    echo ""
    echo -e "Para actualizar, ejecuta: ${GREEN}./update.sh${NC}"
fi
