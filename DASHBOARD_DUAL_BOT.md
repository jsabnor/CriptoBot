# Dashboard - Guía para Mostrar Ambos Bots

## 🎯 Objetivo

Modificar el dashboard para mostrar el estado de ambos bots (ADX y EMA) simultáneamente.

## 📊 Cambios Necesarios

### 1. Backend (`dashboard.py`)

Necesitas añadir un endpoint que lea ambos estados:

```python
@app.route('/api/status')
def api_status():
    """Devuelve estado de ambos bots"""
    status = {
        'adx': {},
        'ema': {},
        'combined': {}
    }
    
    # Leer estado bot ADX
    if os.path.exists('bot_state.json'):
        with open('bot_state.json', 'r') as f:
            status['adx'] = json.load(f)
    
    # Leer estado bot EMA
    if os.path.exists('bot_state_ema.json'):
        with open('bot_state_ema.json', 'r') as f:
            status['ema'] = json.load(f)
    
    # Calcular totales combinados
    adx_equity = status['adx'].get('total_equity', 0)
    ema_equity = status['ema'].get('total_equity', 0)
    
    status['combined'] = {
        'total_equity': adx_equity + ema_equity,
        'adx_equity': adx_equity,
        'ema_equity': ema_equity,
        'adx_percentage': (adx_equity / (adx_equity + ema_equity) * 100) if (adx_equity + ema_equity) > 0 else 0,
        'ema_percentage': (ema_equity / (adx_equity + ema_equity) * 100) if (adx_equity + ema_equity) > 0 else 0
    }
    
    return jsonify(status)
```

### 2. Frontend (`dashboard.html`)

Añadir secciones para mostrar ambos bots:

```html
<!-- Resumen Combinado -->
<div class="metric-card">
    <h3>💰 Equity Total</h3>
    <div id="total-equity">$0.00</div>
    <small>ADX: <span id="adx-equity">$0</span> | EMA: <span id="ema-equity">$0</span></small>
</div>

<!-- Tabs para cada bot -->
<div class="tabs">
    <button class="tab active" onclick="showBot('combined')">Combinado</button>
    <button class="tab" onclick="showBot('adx')">Bot ADX</button>
    <button class="tab" onclick="showBot('ema')">Bot EMA</button>
</div>

<div id="combined-view" class="bot-view">
    <!-- Gráfico comparativo -->
</div>

<div id="adx-view" class="bot-view" style="display:none">
    <!-- Métricas bot ADX -->
</div>

<div id="ema-view" class="bot-view" style="display:none">
    <!-- Métricas bot EMA -->
</div>
```

### 3. JavaScript (`dashboard.js`)

Actualizar para manejar ambos bots:

```javascript
async function updateDashboard() {
    const response = await fetch('/api/status');
    const data = await response.json();
    
    // Actualizar totales combinados
    document.getElementById('total-equity').textContent = 
        `$${data.combined.total_equity.toFixed(2)}`;
    document.getElementById('adx-equity').textContent = 
        `$${data.combined.adx_equity.toFixed(2)}`;
    document.getElementById('ema-equity').textContent = 
        `$${data.combined.ema_equity.toFixed(2)}`;
    
    // Actualizar vistas individuales
    updateBotView('adx', data.adx);
    updateBotView('ema', data.ema);
    updateCombinedView(data.combined);
}

function showBot(botType) {
    // Ocultar todas las vistas
    document.querySelectorAll('.bot-view').forEach(view => {
        view.style.display = 'none';
    });
    
    // Mostrar vista seleccionada
    document.getElementById(`${botType}-view`).style.display = 'block';
    
    // Actualizar tabs activos
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
}
```

## 🎨 Diseño Sugerido

### Layout Propuesto

```
┌─────────────────────────────────────────────────┐
│  Trading Bot Dashboard v1.8.1                   │
├─────────────────────────────────────────────────┤
│  💰 Equity Total: $200.00                       │
│  ADX: $100.00 (50%) | EMA: $100.00 (50%)       │
├─────────────────────────────────────────────────┤
│  [Combinado] [Bot ADX] [Bot EMA]                │
├─────────────────────────────────────────────────┤
│                                                 │
│  📊 Gráfico de Performance                      │
│  ┌───────────────────────────────────────┐     │
│  │ ─── ADX Strategy                      │     │
│  │ ─── EMA Strategy                      │     │
│  │ ─── Combined                          │     │
│  └───────────────────────────────────────┘     │
│                                                 │
│  📈 Métricas                                    │
│  ┌──────────┬──────────┬──────────┐           │
│  │   ADX    │   EMA    │ Combined │           │
│  ├──────────┼──────────┼──────────┤           │
│  │ ROI: 5%  │ ROI: 8%  │ ROI: 6.5%│           │
│  │ Trades:3 │ Trades:2 │ Trades:5 │           │
│  └──────────┴──────────┴──────────┘           │
└─────────────────────────────────────────────────┘
```

## 📝 Implementación Paso a Paso

### Paso 1: Modificar Backend

```bash
# Editar dashboard.py
nano dashboard.py

# Añadir endpoint /api/status como se mostró arriba
# Modificar endpoint /api/trades para incluir ambos CSVs
```

### Paso 2: Actualizar Frontend

```bash
# Editar templates/dashboard.html
nano templates/dashboard.html

# Añadir tabs y vistas para cada bot
```

### Paso 3: Actualizar JavaScript

```bash
# Editar static/dashboard.js
nano static/dashboard.js

# Añadir funciones para manejar múltiples bots
```

### Paso 4: Probar Localmente

```bash
# Ejecutar dashboard
python dashboard.py

# Abrir en navegador
# http://localhost:5000
```

## 🔄 Alternativa Simple (Sin Modificar Dashboard)

Si prefieres no modificar el dashboard ahora, puedes:

### Opción A: Dos Dashboards Separados

Crear `dashboard_ema.py` en puerto diferente:

```python
# dashboard_ema.py
from dashboard import app

# Cambiar archivos de estado
STATE_FILE = 'bot_state_ema.json'
TRADES_FILE = 'trades_ema.csv'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
```

Acceder a:
- Dashboard ADX: `http://localhost:5000`
- Dashboard EMA: `http://localhost:5001`

### Opción B: Script de Comparación

Crear script simple para comparar:

```python
# compare_bots.py
import json

# Leer estados
with open('bot_state.json') as f:
    adx = json.load(f)
with open('bot_state_ema.json') as f:
    ema = json.load(f)

# Mostrar comparación
print("="*50)
print("COMPARACIÓN DE BOTS")
print("="*50)
print(f"ADX Equity: ${adx.get('total_equity', 0):.2f}")
print(f"EMA Equity: ${ema.get('total_equity', 0):.2f}")
print(f"Total: ${adx.get('total_equity', 0) + ema.get('total_equity', 0):.2f}")
```

## 🎯 Recomendación

**Para empezar (Fase Paper Trading):**
- Usa Opción B (Script de Comparación)
- Monitorea ambos bots por separado
- Valida que funcionan correctamente

**Para producción (Después de validar):**
- Implementa dashboard unificado completo
- Añade gráficos comparativos
- Muestra métricas combinadas

## 📊 Métricas a Mostrar

### Vista Combinada
- Equity total
- ROI promedio ponderado
- Total de trades
- Win rate combinado
- Distribución de capital (% ADX vs % EMA)

### Vista Individual (ADX/EMA)
- Equity del bot
- ROI del bot
- Trades del bot
- Win rate del bot
- Posiciones abiertas
- Último trade

## ✅ Próximos Pasos

1. **Ahora**: Usar script de comparación simple
2. **Después de 1 mes**: Evaluar si vale la pena dashboard unificado
3. **Si ambos bots funcionan bien**: Implementar dashboard completo
4. **Si solo uno funciona**: Mantener dashboard actual

---

**Nota:** El dashboard unificado es opcional. Los bots funcionan perfectamente sin él. Puedes monitorizarlos por separado con los archivos JSON y CSV.
