# config.py

# --- Canlı Bot Ayarları ---
LIVE_BOT_SCAN_INTERVAL_MINUTES = 15

# --- Deney ve Simülasyon Ayarları ---
MARKET_ENVIRONMENTS = [
    #{'symbol': 'BTCUSDT', 'timeframe': '1h'},
    {'symbol': 'ETHUSDT', 'timeframe': '1h'},
#    {'symbol': 'BNBUSDT', 'timeframe': '15m'},
#    {'symbol': 'SOLUSDT', 'timeframe': '15m'},
   # {'symbol': 'AVAXUSDT', 'timeframe': '15m'},
  #  {'symbol': 'XRPUSDT', 'timeframe': '15m'},
  #  {'symbol': 'ADAUSDT', 'timeframe': '15m'},
   # {'symbol': 'MATICUSDT', 'timeframe': '15m'},
    #{'symbol': 'ARBUSDT', 'timeframe': '15m'},
]
TOTAL_DAYS = 180
TRAINING_DAYS = 120

# --- Portföy ve Risk Ayarları ---
INITIAL_CAPITAL = 10000.0
CAPITAL_PER_TRADE_PERCENT = 0.25
MAX_CONCURRENT_POSITIONS = 4

# --- Risk Yönetimi Ayarları ---
USE_STOP_LOSS = True
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

# --- YENİ: Piyasa Rejimi Ayarları ---
REGIME_DETECTION_PERIOD = 200 # Piyasa trendini belirlemek için kullanılacak yavaş hareketli ortalama periyodu

# --- Strateji Sentezleyici Ayarları ---
ACTIVE_INDICATORS = {
    'rsi': True,
    'macd': True,
    'ma_cross': True,
    'bbands': True,
    'obv': True,
    'ott': True,  # YENİ: OTT göstergesini aktif ettik
}

# --- GENİŞLETİLMİŞ PARAMETRE UZAYLARI ---
# Ajanın yeni stratejiler üretirken kullanacağı parametre aralıkları
PARAMETER_SPACES = {
    'RSI': {
        'rsi_period': [7, 14, 21]  # Hızlı, standart ve yavaş RSI
    },
    'MA_CROSS': {
        'fast_period': [10, 20, 50],   # Kısa, orta ve uzun vadeli hızlı ortalamalar
        'slow_period': [30, 50, 100, 200] # Kısa, orta ve uzun vadeli yavaş ortalamalar
    },
    'BBANDS': {
        'length': [20], 
        'std': [2.0, 2.5, 3.0] # Standart, geniş ve çok geniş bantlar
    },
    'MACD': [
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, # Standart
        {'fast_period': 5, 'slow_period': 35, 'signal_period': 5}   # Daha hızlı bir MACD
    ],
    'OBV': [
        {'obv_sma_period': 20},
        {'obv_sma_period': 50}
    ],
    'OTT': {
        'length': [2, 5],
        'percent': [1.4, 2.0, 3.0] # Farklı hassasiyetlerde OTT
    }
}