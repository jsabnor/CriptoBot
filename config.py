import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# CONFIGURACIÓN CENTRALIZADA - BOT DE TRADING
# ============================================================================

# Credenciales API
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Configuración General
TRADING_MODE = os.getenv('TRADING_MODE', 'paper').lower()
CAPITAL_PER_PAIR = float(os.getenv('CAPITAL_PER_PAIR', '50.0'))
TIMEFRAME = os.getenv('TIMEFRAME', '4h')

# Pares de Trading (Lista separada por comas en .env o default)
_symbols_env = os.getenv('SYMBOLS')
if _symbols_env:
    SYMBOLS = [s.strip() for s in _symbols_env.split(',')]
else:
    # Top 3 performers basados en backtest (excluye BNB)
    SYMBOLS = ['SOL/USDT', 'ETH/USDT', 'XRP/USDT']

# ============================================================================
# ESTRATEGIA (v1.6.0 OPTIMIZED)
# ============================================================================
# Estos valores pueden ser sobrescritos desde el archivo .env

# Gestión de Riesgo
RISK_PERCENT = float(os.getenv('RISK_PERCENT', '0.04'))        # 4% por operación
MIN_EQUITY = float(os.getenv('MIN_EQUITY', '10.0'))            # Capital mínimo libre
MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '2')) # Máx trades diarios por par
COMMISSION = float(os.getenv('COMMISSION', '0.001'))           # Comisión Binance (0.1%)

# Indicadores Técnicos
ATR_LENGTH = int(os.getenv('ATR_LENGTH', '14'))
ATR_MULTIPLIER = float(os.getenv('ATR_MULTIPLIER', '4.0'))     # Stop Loss amplio
MA_LENGTH = int(os.getenv('MA_LENGTH', '50'))                  # Tendencia media
LONG_MA_LENGTH = int(os.getenv('LONG_MA_LENGTH', '200'))       # Tendencia largo plazo
ADX_LENGTH = int(os.getenv('ADX_LENGTH', '14'))
ADX_THRESHOLD = int(os.getenv('ADX_THRESHOLD', '30'))          # Filtro de fuerza de tendencia
TRAILING_TP_PERCENT = float(os.getenv('TRAILING_TP_PERCENT', '0.50')) # Trailing Stop 50%
