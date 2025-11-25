#!/usr/bin/env python3
"""
Script simple para comparar el rendimiento de ambos bots
Uso: python compare_bots.py
"""

import json
import os
from datetime import datetime

def load_state(filename):
    """Carga el estado de un bot"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def main():
    # Cargar estados
    adx_state = load_state('bot_state.json')
    ema_state = load_state('bot_state_ema.json')
    
    print("\n" + "="*60)
    print(" " * 15 + "COMPARACIÓN DE BOTS")
    print("="*60 + "\n")
    
    if not adx_state and not ema_state:
        print("❌ No se encontraron archivos de estado")
        print("   Asegúrate de que los bots hayan ejecutado al menos un ciclo")
        return
    
    # Bot ADX
    if adx_state:
        print("📊 BOT ADX (Estrategia ADX + ATR + MA)")
        print("-" * 60)
        adx_equity = adx_state.get('total_equity', 0)
        adx_positions = sum(1 for pos in adx_state.get('positions', {}).values() if pos)
        print(f"  Equity Total:      ${adx_equity:.2f}")
        print(f"  Posiciones Abiertas: {adx_positions}")
        print(f"  Última Actualización: {adx_state.get('timestamp', 'N/A')}")
    else:
        print("📊 BOT ADX: No disponible")
        adx_equity = 0
    
    print()
    
    # Bot EMA
    if ema_state:
        print("📈 BOT EMA (Estrategia EMA Crossover)")
        print("-" * 60)
        ema_equity = ema_state.get('equity', {})
        ema_total = sum(ema_equity.values()) if isinstance(ema_equity, dict) else 0
        ema_positions = sum(1 for pos in ema_state.get('positions', {}).values() if pos)
        print(f"  Equity Total:      ${ema_total:.2f}")
        print(f"  Posiciones Abiertas: {ema_positions}")
        print(f"  Última Actualización: {ema_state.get('last_update', 'N/A')}")
    else:
        print("📈 BOT EMA: No disponible")
        ema_total = 0
    
    print()
    print("="*60)
    print("💰 RESUMEN COMBINADO")
    print("="*60)
    
    total_equity = adx_equity + ema_total
    
    if total_equity > 0:
        adx_percentage = (adx_equity / total_equity) * 100
        ema_percentage = (ema_total / total_equity) * 100
        
        print(f"  Equity Total:      ${total_equity:.2f}")
        print(f"  Bot ADX:           ${adx_equity:.2f} ({adx_percentage:.1f}%)")
        print(f"  Bot EMA:           ${ema_total:.2f} ({ema_percentage:.1f}%)")
        
        # Calcular ROI si hay capital inicial
        initial_capital = 200  # Ajustar según tu configuración
        roi = ((total_equity - initial_capital) / initial_capital) * 100
        print(f"  ROI Total:         {roi:+.2f}%")
    else:
        print("  No hay datos de equity disponibles")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
