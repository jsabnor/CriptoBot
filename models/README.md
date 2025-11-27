# Neural Strategy - Modelos entrenados

Este directorio contiene los modelos entrenados de la estrategia neuronal.

## Estructura

```
models/
├── neural_model_v1.keras      # Modelo versión 1
├── scaler_v1.pkl               # Scaler de normalización v1
├── metrics_v1.json             # Métricas de rendimiento v1
├── checkpoints/                # Checkpoints durante entrenamiento
└── logs/                       # Logs y resultados de backtest
```

## Archivos

- **neural_model_vX.keras**: Modelo de red neuronal entrenado
- **scaler_vX.pkl**: Scaler MinMaxScaler para normalización de features
- **metrics_vX.json**: Métricas de evaluación del modelo

## Versionado

El sistema mantiene automáticamente las últimas 5 versiones de modelos.
Cada reentrenamiento incrementa el número de versión.

## No subir a Git

Este directorio está en `.gitignore` ya que los modelos son grandes y específicos de cada instalación.
