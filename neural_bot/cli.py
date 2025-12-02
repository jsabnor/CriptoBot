"""
Neural Bot CLI - Interfaz de línea de comandos para gestión del sistema neural

Comandos disponibles:
    list            - Lista modelos disponibles
    info            - Información detallada de un modelo
    set-default     - Establece modelo por defecto
    delete          - Elimina un modelo
    train           - Entrena nuevo modelo
    backtest        - Ejecuta backtest con un modelo
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Tabulate es opcional - fallback a formato simple
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Añadir directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from neural_bot import ModelManager, NeuralStrategy, NeuralBacktest
from neural_bot.config import config
from neural_bot.strategy import ContinuousLearner


def cmd_list(args):
    """Lista todos los modelos disponibles"""
    manager = ModelManager()
    models = manager.list_models()
    
    if not models:
        print("📦 No hay modelos guardados")
        return
    
    # Preparar tabla
    table_data = []
    for model in models:
        row = [
            '✓' if model['is_default'] else '',
            model['name'],
            model.get('accuracy', 'N/A'),
            ', '.join(model.get('symbols', [])) if model.get('symbols') else 'N/A',
            model.get('timeframe', 'N/A'),
            datetime.fromisoformat(model['created_at']).strftime('%Y-%m-%d %H:%M')
        ]
        table_data.append(row)
    
    headers = ['Default', 'Name', 'Accuracy', 'Symbols', 'TF', 'Created']
    print("\n📦 Modelos Disponibles:\n")
    
    if HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        # Fallback simple sin tabulate
        print(" | ".join(headers))
        print("-" * 80)
        for row in table_data:
            print(" | ".join(str(cell) for cell in row))
    print()


def cmd_info(args):
    """Muestra información detallada de un modelo"""
    manager = ModelManager()
    metadata = manager.get_model_metadata(args.model)
    
    if metadata is None:
        print(f"❌ Modelo '{args.model}' no encontrado")
        return
    
    print(f"\n📊 Información del Modelo: {args.model}\n")
    print(f"{'='*60}")
    
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"{key:20s}: {', '.join(map(str, value))}")
        elif isinstance(value, dict):
            print(f"{key:20s}:")
            for k, v in value.items():
                print(f"  {k:18s}: {v}")
        else:
            print(f"{key:20s}: {value}")
    
    print(f"{'='*60}\n")


def cmd_set_default(args):
    """Establece un modelo como predeterminado"""
    manager = ModelManager()
    if manager.set_default_model(args.model):
        print(f"✅ Modelo '{args.model}' establecido como predeterminado")
    else:
        print(f"❌ No se pudo establecer '{args.model}' como predeterminado")


def cmd_delete(args):
    """Elimina un modelo"""
    manager = ModelManager()
    
    if not args.force:
        confirm = input(f"¿Seguro que quieres eliminar '{args.model}'? (s/N): ")
        if confirm.lower() != 's':
            print("❌ Operación cancelada")
            return
    
    if manager.delete_model(args.model, force=args.force):
        print(f"✅ Modelo '{args.model}' eliminado")
    else:
        print(f"❌ No se pudo eliminar '{args.model}'")


def cmd_train(args):
    """Entrena un nuevo modelo"""
    print(f"\n🎓 Entrenando nuevo modelo: {args.name}\n")
    
    # Preparar símbolos
    symbols = args.symbols.split(',') if args.symbols else config.DEFAULT_SYMBOLS
    
    # Mostrar configuración
    if args.start_date or args.end_date:
        print(f"📅 Período de entrenamiento:")
        print(f"   Desde: {args.start_date or 'inicio disponible'}")
        print(f"   Hasta: {args.end_date or 'presente'}\n")
    
    # Entrenar
    learner = ContinuousLearner()
    model = learner.train_initial_model(
        symbols=symbols, 
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if model is None:
        print("❌ Error durante el entrenamiento")
        return
    
    # Guardar con nombre personalizado
    manager = ModelManager()
    metadata = {
        'symbols': symbols,
        'timeframe': args.timeframe,
        'description': args.description or f"Model trained on {', '.join(symbols)}",
        'accuracy': learner.metrics_history[-1].get('accuracy') if learner.metrics_history else None,
    }
    
    if manager.save_model(model.model, learner.feature_extractor.scaler, args.name, metadata):
        print(f"\n✅ Modelo guardado como: {args.name}")
        
        # Marcar como default si se especificó
        if args.set_default:
            manager.set_default_model(args.name)
            print(f"✅ '{args.name}' establecido como modelo por defecto")


def cmd_backtest(args):
    """Ejecuta backtest con un modelo específico"""
    print(f"\n📈 Ejecutando backtest con modelo: {args.model or 'default'}\n")
    
    # Cargar estrategia con modelo específico
    strategy = NeuralStrategy(model_name=args.model)
    
    if strategy.model is None:
        print("❌ No se pudo cargar el modelo")
        return
    
    # Ejecutar backtest
    backtester = NeuralBacktest(capital_per_pair=args.capital)
    
    if args.symbol:
        # Backtest en un solo símbolo
        results = backtester.backtest_symbol(
            args.symbol,
            strategy,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"\n✅ Backtest completado para {args.symbol}")
    else:
        # Backtest en múltiples símbolos
        symbols = args.symbols.split(',') if args.symbols else config.DEFAULT_SYMBOLS
        results = backtester.backtest_multiple(
            symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"\n✅ Backtest completado para {len(symbols)} símbolos")


def main():
    parser = argparse.ArgumentParser(
        description='Neural Bot CLI - Gestión del sistema de trading neural',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: list
    parser_list = subparsers.add_parser('list', help='Lista modelos disponibles')
    parser_list.set_defaults(func=cmd_list)
    
    # Comando: info
    parser_info = subparsers.add_parser('info', help='Información de un modelo')
    parser_info.add_argument('model', help='Nombre del modelo')
    parser_info.set_defaults(func=cmd_info)
    
    # Comando: set-default
    parser_default = subparsers.add_parser('set-default', help='Establece modelo por defecto')
    parser_default.add_argument('model', help='Nombre del modelo')
    parser_default.set_defaults(func=cmd_set_default)
    
    # Comando: delete
    parser_delete = subparsers.add_parser('delete', help='Elimina un modelo')
    parser_delete.add_argument('model', help='Nombre del modelo')
    parser_delete.add_argument('--force', action='store_true', help='Forzar eliminación sin confirmación')
    parser_delete.set_defaults(func=cmd_delete)
    
    # Comando: train
    parser_train = subparsers.add_parser('train', help='Entrena nuevo modelo')
    parser_train.add_argument('--name', required=True, help='Nombre del modelo')
    parser_train.add_argument('--symbols', help='Símbolos separados por comas (ej: ETH/USDT,SOL/USDT)')
    parser_train.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser_train.add_argument('--start-date', help='Fecha inicio datos (YYYY-MM-DD)')
    parser_train.add_argument('--end-date', help='Fecha fin datos (YYYY-MM-DD)')
    parser_train.add_argument('--description', help='Descripción del modelo')
    parser_train.add_argument('--set-default', action='store_true', help='Marcar como modelo por defecto')
    parser_train.set_defaults(func=cmd_train)
    
    # Comando: backtest
    parser_backtest = subparsers.add_parser('backtest', help='Ejecuta backtest')
    parser_backtest.add_argument('--model', help='Nombre del modelo (default: usa el predeterminado)')
    parser_backtest.add_argument('--symbol', help='Símbolo único para backtest')
    parser_backtest.add_argument('--symbols', help='Símbolos separados por comas')
    parser_backtest.add_argument('--start-date', help='Fecha inicial (YYYY-MM-DD)')
    parser_backtest.add_argument('--end-date', help='Fecha final (YYYY-MM-DD)')
    parser_backtest.add_argument('--capital', type=float, default=50, help='Capital por par (default: 50)')
    parser_backtest.set_defaults(func=cmd_backtest)
    
    # Parse argumentos
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Ejecutar comando
    try:
        args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
