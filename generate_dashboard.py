import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# ============================================================================
# DASHBOARD DE VISUALIZACIÓN - BOT v1.0 PRODUCTION
# ============================================================================
# Genera gráficas de rendimiento del bot production
# ============================================================================

def load_backtest_results():
    """Carga resultados del backtesting multi-activo."""
    if os.path.exists('backtest_multi_results.csv'):
        return pd.read_csv('backtest_multi_results.csv')
    return None

def create_roi_comparison():
    """Crea gráfica comparativa de ROI por par y timeframe."""
    df = load_backtest_results()
    if df is None:
        print("No se encontraron resultados de backtesting")
        return
    
    # Filtrar los mejores (4h)
    df_4h = df[df['timeframe'] == '4h'].sort_values('total_return', ascending=False)
    
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_4h['total_return']]
    bars = plt.bar(df_4h['symbol'], df_4h['total_return'], color=colors, edgecolor='black', linewidth=1.5)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=10)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('📊 ROI por Criptomoneda (Timeframe 4h)\nBacktesting 2020-2025', fontsize=14, fontweight='bold')
    plt.xlabel('Par', fontsize=12)
    plt.ylabel('ROI (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('roi_comparison_4h.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: roi_comparison_4h.png")
    plt.close()

def create_timeframe_comparison():
    """Crea comparación por timeframe."""
    df = load_backtest_results()
    if df is None:
        return
    
    # Agrupar por timeframe
    tf_stats = df.groupby('timeframe').agg({
        'total_return': 'mean',
        'win_rate': 'mean',
        'sharpe_ratio': 'mean',
        'total_trades': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📈 Comparación por Timeframe (Promedio de 7 pares)', fontsize=16, fontweight='bold')
    
    # ROI
    axes[0, 0].bar(tf_stats['timeframe'], tf_stats['total_return'], color=['#e74c3c', '#2ecc71', '#3498db'])
    axes[0, 0].set_title('ROI Promedio (%)', fontweight='bold')
    axes[0, 0].set_ylabel('ROI (%)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Win Rate
    axes[0, 1].bar(tf_stats['timeframe'], tf_stats['win_rate'], color=['#e74c3c', '#2ecc71', '#3498db'])
    axes[0, 1].set_title('Win Rate Promedio (%)', fontweight='bold')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Sharpe Ratio
    axes[1, 0].bar(tf_stats['timeframe'], tf_stats['sharpe_ratio'], color=['#e74c3c', '#2ecc71', '#3498db'])
    axes[1, 0].set_title('Sharpe Ratio Promedio', fontweight='bold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Número de Trades
    axes[1, 1].bar(tf_stats['timeframe'], tf_stats['total_trades'], color=['#e74c3c', '#2ecc71', '#3498db'])
    axes[1, 1].set_title('Trades Promedio (5 años)', fontweight='bold')
    axes[1, 1].set_ylabel('Número de Trades')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeframe_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: timeframe_comparison.png")
    plt.close()

def create_heatmap():
    """Crea heatmap de ROI por par y timeframe."""
    df = load_backtest_results()
    if df is None:
        return
    
    # Pivot table
    pivot = df.pivot(index='symbol', columns='timeframe', values='total_return')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'ROI (%)'}, linewidths=1, linecolor='black')
    plt.title('🔥 Heatmap de ROI: Par vs Timeframe\n(Backtesting 2020-2025)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Timeframe', fontsize=12)
    plt.ylabel('Par', fontsize=12)
    plt.tight_layout()
    plt.savefig('roi_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: roi_heatmap.png")
    plt.close()

def create_top_configs():
    """Muestra top 10 configuraciones."""
    df = load_backtest_results()
    if df is None:
        return
    
    top10 = df.nlargest(10, 'total_return')[['symbol', 'timeframe', 'total_return', 'win_rate', 'sharpe_ratio', 'total_trades']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    table_data = []
    for i, row in top10.iterrows():
        table_data.append([
            f"{row['symbol']}",
            row['timeframe'],
            f"{row['total_return']:.1f}%",
            f"{row['win_rate']:.1f}%",
            f"{row['sharpe_ratio']:.3f}",
            f"{int(row['total_trades'])}"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Par', 'TF', 'ROI', 'Win Rate', 'Sharpe', 'Trades'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.1, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo
    for i in range(len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.title('🏆 Top 10 Configuraciones por ROI\n(Backtesting 2020-2025)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('top10_configs.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfica guardada: top10_configs.png")
    plt.close()

def generate_dashboard():
    """Genera todas las visualizaciones."""
    print("\n" + "="*70)
    print("📊 GENERANDO DASHBOARD DE VISUALIZACIÓN")
    print("="*70 + "\n")
    
    create_roi_comparison()
    create_timeframe_comparison()
    create_heatmap()
    create_top_configs()
    
    print("\n" + "="*70)
    print("✅ DASHBOARD COMPLETADO")
    print("="*70)
    print("\nGráficas generadas:")
    print("  1. roi_comparison_4h.png - Comparación ROI en 4h")
    print("  2. timeframe_comparison.png - Estadísticas por timeframe")
    print("  3. roi_heatmap.png - Heatmap par vs timeframe")
    print("  4. top10_configs.png - Top 10 configuraciones")
    print("\n")

if __name__ == "__main__":
    generate_dashboard()
