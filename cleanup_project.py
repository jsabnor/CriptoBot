import os
import shutil
from pathlib import Path

# ============================================================================
# SCRIPT DE LIMPIEZA - Organiza el proyecto dejando solo lo esencial
# ============================================================================

def cleanup_project():
    """Limpia el proyecto dejando solo archivos esenciales."""
    
    base_dir = Path.cwd()
    
    # Crear carpeta para archivos antiguos
    archive_dir = base_dir / "archive_old_versions"
    archive_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("🧹 LIMPIEZA DE PROYECTO")
    print("="*70 + "\n")
    
    # Archivos ESENCIALES (mantener en raíz)
    keep_files = {
        'bot_production.py',           # Bot principal
        'backtest_multi.py',           # Sistema de backtesting
        'generate_dashboard.py',       # Visualizaciones
        'backtest_multi_results.csv',  # Resultados backtesting
        'roi_comparison_4h.png',       # Gráfica 1
        'timeframe_comparison.png',    # Gráfica 2
        'roi_heatmap.png',             # Gráfica 3
        'top10_configs.png',           # Gráfica 4
    }
    
    # Archivos a ARCHIVAR (mover a carpeta archive)
    archive_patterns = [
        'bot_v0.',      # Versiones antiguas 0.x
        'bot_v1.',      # Versiones de desarrollo 1.x
        'bot.py',       # Bot original
        'bot.txt',
        'analisis_log',
        'equity_curve_v',   # Logs antiguos
        'equity_curve.csv',
        'trades_log_v',     # Trades antiguos
        'trades_log.csv',
        'metrics_v',        # Métricas antiguas
    ]
    
    # Contar archivos
    moved = 0
    kept = 0
    
    # Procesar archivos
    for file_path in base_dir.iterdir():
        if file_path.is_file():
            file_name = file_path.name
            
            # Skip si es este script
            if file_name == 'cleanup_project.py':
                continue
            
            # Mantener archivos esenciales
            if file_name in keep_files:
                kept += 1
                print(f"✅ MANTENER: {file_name}")
                continue
            
            # Archivar si coincide con patrones
            should_archive = any(pattern in file_name for pattern in archive_patterns)
            
            if should_archive:
                dest = archive_dir / file_name
                shutil.move(str(file_path), str(dest))
                moved += 1
                print(f"📦 ARCHIVADO: {file_name} → archive_old_versions/")
    
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    print(f"✅ Archivos mantenidos: {kept}")
    print(f"📦 Archivos archivados: {moved}")
    print(f"📁 Ubicación archivo: {archive_dir}")
    print("="*70 + "\n")
    
    # Mostrar estructura final
    print("📂 ESTRUCTURA FINAL DEL PROYECTO:")
    print("="*70)
    print("\nbot/ (directorio raíz)")
    print("│")
    print("├── 🚀 BOT PRINCIPAL")
    print("│   └── bot_production.py")
    print("│")
    print("├── 🧪 BACKTESTING")
    print("│   ├── backtest_multi.py")
    print("│   └── backtest_multi_results.csv")
    print("│")
    print("├── 📊 VISUALIZACIÓN")
    print("│   ├── generate_dashboard.py")
    print("│   ├── roi_comparison_4h.png")
    print("│   ├── timeframe_comparison.png")
    print("│   ├── roi_heatmap.png")
    print("│   └── top10_configs.png")
    print("│")
    print("├── 📁 DATA (carpeta de datos)")
    print("│   └── [velas históricas]")
    print("│")
    print("└── 📦 ARCHIVE (versiones antiguas)")
    print("    └── [bot_v0.x, bot_v1.x, logs antiguos]")
    print("="*70 + "\n")
    
    print("✅ LIMPIEZA COMPLETADA\n")
    print("📚 DOCUMENTACIÓN:")
    print("   Los archivos de documentación (README, guías) están en:")
    print("   ~/.gemini/antigravity/brain/.../artifacts/\n")
    
    print("🚀 PRÓXIMO PASO:")
    print("   python bot_production.py\n")

if __name__ == "__main__":
    cleanup_project()
