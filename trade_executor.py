import os
import math
import time
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

try:
    import pandas as pd
except ImportError:
    pd = None

load_dotenv()
DRY_RUN = False
binance_api_key, binance_api_secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")

if not binance_api_key or not binance_api_secret:
    print("HATA: Binance API anahtarları .env dosyasında bulunamadı."); client = None
else:
    client = Client(binance_api_key, binance_api_secret); client.API_URL = 'https://testnet.binance.vision/api'

SYMBOL_RULES = {}

def get_rules_for_symbol(symbol: str):
    if symbol in SYMBOL_RULES: return SYMBOL_RULES[symbol]
    for attempt in range(3):
        try:
            print(f"'{symbol}' için borsa kuralları çekiliyor...")
            info = client.get_symbol_info(symbol)
            if not info: return None
            rules = {}
            for f in info.get('filters', []):
                if f['filterType'] == 'LOT_SIZE':
                    rules['stepSize'], rules['minQty'] = float(f['stepSize']), float(f['minQty'])
            SYMBOL_RULES[symbol] = rules
            return rules
        except Exception as e:
            print(f"Hata: '{symbol}' için kurallar çekilemedi (Deneme {attempt+1}/3) - {e}")
            if attempt < 2: time.sleep(5)
    return None

def adjust_quantity_to_rules(quantity: float, rules: dict) -> float:
    min_qty = rules.get('minQty', 0.0)
    if quantity < min_qty:
        print(f"Uyarı: Miktar ({quantity}), minimumdan ({min_qty}) az."); return 0.0
    step_size = rules.get('stepSize', 0.000001)
    adjusted_quantity = math.floor(quantity / step_size) * step_size
    precision = abs(int(round(math.log10(step_size), 0)))
    return round(adjusted_quantity, precision)

def get_asset_balance(asset: str) -> float:
    if not client: return 0.0
    for attempt in range(3):
        try:
            balance_info = client.get_asset_balance(asset=asset)
            return float(balance_info['free'])
        except Exception as e:
            print(f"'{asset}' bakiyesi çekilirken hata (Deneme {attempt+1}/3) - {e}")
            if attempt < 2: time.sleep(5)
    return 0.0

def place_order(symbol: str, side: str, quantity: float, order_type: str = 'MARKET'):
    if not client: return None
    print("-" * 30); print(f"Ham Emir: {side} {quantity:.8f} {symbol}")
    rules = get_rules_for_symbol(symbol)
    if not rules: return None
    adjusted_quantity = adjust_quantity_to_rules(quantity, rules)
    if adjusted_quantity <= 0: return None
    print(f"Ayarlanmış Emir: {side} {adjusted_quantity:.8f} {symbol}")
    if DRY_RUN:
        print("### DRY RUN MODU ###"); return {'status': 'FILLED'}
    else:
        for attempt in range(3):
            try:
                print(">>> TESTNET'E GERÇEK EMİR GÖNDERİLİYOR! <<<")
                order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=adjusted_quantity)
                print("Emir başarıyla gönderildi:"); print(order)
                return order
            except Exception as e:
                print(f"API Hatası (Deneme {attempt+1}/3): Emir gönderilemedi. Hata: {e}")
                if attempt < 2: time.sleep(5)
    return None

if __name__ == "__main__":
    if pd is None: print("Pandas gerekli ('pip install pandas')."); exit()
    print("Bakiye Kontrol Modülü Başlatıldı.")
    if not client: print("Binance istemcisi başlatılamadı.")
    else:
        try:
            account_info = client.get_account()
            balances = account_info.get('balances', [])
            print("\n--- TESTNET CÜZDAN BAKİYELERİ ---")
            found_assets = False
            for asset in balances:
                if float(asset['free']) > 0:
                    print(f"- Varlık: {asset['asset']}, Bakiye: {asset['free']}")
                    found_assets = True
            if not found_assets: print("Cüzdanda herhangi bir varlık bulunamadı.")
        except Exception as e: print(f"Bakiyeler alınırken bir hata oluştu: {e}")