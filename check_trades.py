import os
from dotenv import load_dotenv
from binance.client import Client
from config import MARKET_ENVIRONMENTS # Artık config'den hangi varlıkları izlediğimizi alıyoruz

# Pandas kütüphanesini en başta import edip kontrol edelim
try:
    import pandas as pd
except ImportError:
    print("Pandas kütüphanesi bulunamadı. Lütfen 'pip install pandas' ile kurun.")
    exit()

load_dotenv()

# API anahtarlarını ve istemciyi yapılandır
binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")

if not binance_api_key or not binance_api_secret:
    print("HATA: Binance API anahtarları .env dosyasında bulunamadı.")
    client = None
else:
    client = Client(binance_api_key, binance_api_secret)
    # Testnet'e bağlandığımızdan emin olalım
    client.API_URL = 'https://testnet.binance.vision/api'

def fetch_my_trades(symbol: str) -> list:
    """Belirtilen parite için Testnet'teki işlem geçmişini bir liste olarak döndürür."""
    if not client:
        print("Binance istemcisi başlatılamadı.")
        return []

    print(f"'{symbol}' için işlem geçmişi çekiliyor...")
    try:
        # Hesabımıza ait işlemleri çekme komutu
        trades = client.get_my_trades(symbol=symbol, limit=50) # Her parite için son 50 işlemi alalım
        return trades
    except Exception as e:
        print(f"'{symbol}' için işlem geçmişi alınırken bir hata oluştu: {e}")
        return []

if __name__ == "__main__":
    # Config'den portföydeki benzersiz sembollerin bir listesini oluştur
    symbols_to_check = list(set(market['symbol'] for market in MARKET_ENVIRONMENTS))
    
    all_trades = []
    # Her bir sembol için işlem geçmişini çek ve tek bir listede topla
    for symbol in symbols_to_check:
        trades_for_symbol = fetch_my_trades(symbol)
        all_trades.extend(trades_for_symbol)
        
    if not all_trades:
        print("\nİzlenen varlıkların hiçbirinde herhangi bir işlem bulunamadı.")
    else:
        # Tüm işlemleri zamana göre en yeniden en eskiye doğru sırala
        all_trades.sort(key=lambda x: x['time'], reverse=True)
        
        print("\n--- TÜM PORTFÖY İÇİN SON İŞLEMLER (En Yeni En Üstte) ---")
        for trade in all_trades:
            print("-" * 30)
            print(f"Parite:       {trade['symbol']}")
            print(f"Tarih/Zaman:  {pd.to_datetime(trade['time'], unit='ms')}")
            print(f"Emir Tipi:    {'ALIŞ' if trade['isBuyer'] else 'SATIŞ'}")
            print(f"Miktar:       {trade['qty']}")
            print(f"Fiyat:        {trade['price']}")
            print(f"Komisyon:     {trade['commission']} {trade['commissionAsset']}")
            print(f"İşlem ID:     {trade['id']}")