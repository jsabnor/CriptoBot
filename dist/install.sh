#!/bin/bash

echo "📦 Instalando dependencias..."

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "   - Creando entorno virtual (venv)..."
    python3 -m venv venv
fi

# Activar entorno
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalación
if [ -f "requirements.txt" ]; then
    echo "   - Instalando paquetes desde requirements.txt..."
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt no encontrado."
    exit 1
fi

echo "✅ Instalación completada."
echo "   Ejecuta './start_bots.sh' para iniciar."
