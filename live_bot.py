import os
import json
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client as BinanceClient
import requests

# DİKKAT: Yeni config değişkenini import ediyoruz
from config import (
    MARKET_ENVIRONMENTS, CAPITAL_PER_TRADE_PERCENT, MAX_CONCURRENT_POSITIONS,
    LIVE_BOT_SCAN_INTERVAL_MINUTES
)
from strategy_engine import check_live_signal
from trade_executor import place_order, get_asset_balance

# --- Kurulum ---
load_dotenv()
binance_client = BinanceClient(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
binance_client.API_URL = 'https://testnet.binance.vision/api'
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHAMPIONS_FILE = 'champions.json'
POSITIONS_FILE = 'positions.json'

def send_telegram_notification(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"\nTelegram'a bağlanırken bir hata oluştu: {e}")

def load_positions():
    if not os.path.exists(POSITIONS_FILE): return {}
    try:
        with open(POSITIONS_FILE, 'r') as f: return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError): return {}

def save_positions(positions: dict):
    with open(POSITIONS_FILE, 'w') as f: json.dump(positions, f, indent=4)

def get_open_positions():
    for attempt in range(3):
        try:
            account_info = binance_client.get_account()
            positions = {}
            assets_in_portfolio = [m['symbol'].replace('USDT', '') for m in MARKET_ENVIRONMENTS]
            for asset in account_info.get('balances', []):
                asset_name, free_balance = asset['asset'], float(asset['free'])
                if asset_name in assets_in_portfolio and free_balance > 0.00001:
                     positions[f"{asset_name}USDT"] = {'quantity': free_balance}
            return positions
        except Exception as e:
            print(f"Hata: Açık pozisyonlar alınamadı (Deneme {attempt + 1}/3) - {e}")
            if attempt < 2:
                print("5 saniye sonra tekrar denenecek..."); time.sleep(5)
    return {}

def run_live_bot():
    if not os.path.exists(CHAMPIONS_FILE):
        print(f"HATA: '{CHAMPIONS_FILE}' dosyası bulunamadı. Lütfen önce portfolio_simulator.py'yi çalıştırın.")
        return
    with open(CHAMPIONS_FILE, 'r') as f:
        champion_strategies = json.load(f)
    print("🏆 Şampiyon stratejiler yüklendi.")
    send_telegram_notification("🤖 *Cognitive Trade Agent (Canlı Test) Başlatıldı!*\nŞampiyon stratejiler yüklendi, piyasa taranıyor...")

    # DÜZELTME: Uyku süresini artık doğrudan config'den alıyoruz.
    sleep_duration_seconds = LIVE_BOT_SCAN_INTERVAL_MINUTES * 60

    while True:
        try:
            print("\n" + "="*50)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🤖 Ajan uyanıyor...")
            
            positions_from_file = load_positions()
            positions_from_exchange = get_open_positions()
            
            for symbol in list(positions_from_file.keys()):
                if symbol not in positions_from_exchange:
                    print(f"Senkronizasyon: {symbol} pozisyonu dosyadan silindi (borsada bulunamadı).")
                    del positions_from_file[symbol]
            
            for symbol, details in positions_from_exchange.items():
                if symbol not in positions_from_file:
                    print(f"Senkronizasyon: {symbol} pozisyonu dosyaya eklendi (giriş fiyatı bilinmiyor).")
                    positions_from_file[symbol] = {'quantity': details['quantity'], 'entry_price': 0}
            
            save_positions(positions_from_file)
            open_positions = positions_from_file
            usdt_balance = get_asset_balance('USDT')
            print(f"Mevcut Bakiye: {usdt_balance:.2f} USDT | Açık Pozisyonlar: {list(open_positions.keys())}")

            for market in MARKET_ENVIRONMENTS:
                key = f"{market['symbol']}-{market['timeframe']}"
                if key not in champion_strategies: continue
                klines = binance_client.get_historical_klines(market['symbol'], market['timeframe'], "1 day ago UTC")
                live_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns: live_df[col] = pd.to_numeric(live_df[col])
                live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms')
                
                champion_strategy, signal, current_price = champion_strategies[key], check_live_signal(live_df, champion_strategies[key]), live_df.iloc[-1]['close']
                print(f"-> {key}: Şampiyon = {champion_strategy['name']}, Sinyal = {signal}, Fiyat = {current_price:.2f}")

                if market['symbol'] in open_positions and signal == 'SELL':
                    position_details = open_positions[market['symbol']]
                    quantity_to_sell, entry_price = position_details['quantity'], position_details.get('entry_price', 0)
                    pnl_amount, pnl_percent = ((current_price - entry_price) * quantity_to_sell, ((current_price / entry_price) - 1) * 100) if entry_price > 0 else (0, 0)
                    order_result = place_order(market['symbol'], BinanceClient.SIDE_SELL, round(quantity_to_sell, 5))
                    if order_result:
                        pnl_text = f"✅ *Sonuç: +${pnl_amount:.2f} (+{pnl_percent:.2f}%)*" if pnl_amount > 0 else f"🔻 *Sonuç: ${pnl_amount:.2f} ({pnl_percent:.2f}%)*"
                        if entry_price == 0: pnl_text = "*Sonuç: Bilinmiyor (giriş fiyatı hafızada yok)*"
                        send_telegram_notification(f"🔥 SATIŞ: {quantity_to_sell:.5f} {market['symbol']} @ {current_price:.2f} pozisyonu kapatıldı.\n{pnl_text}")
                        del open_positions[market['symbol']]; save_positions(open_positions)
                
                elif market['symbol'] not in open_positions and signal == 'BUY':
                    if len(open_positions) < MAX_CONCURRENT_POSITIONS:
                        capital_to_use = usdt_balance * CAPITAL_PER_TRADE_PERCENT
                        if capital_to_use > 15:
                             quantity_to_buy = round(capital_to_use / current_price, 5)
                             order_result = place_order(market['symbol'], BinanceClient.SIDE_BUY, quantity_to_buy)
                             if order_result:
                                 send_telegram_notification(f"🚀 ALIŞ: {quantity_to_buy:.5f} {market['symbol']} @ {current_price:.2f} pozisyonu açıldı.")
                                 open_positions[market['symbol']] = {'quantity': quantity_to_buy, 'entry_price': current_price}; save_positions(open_positions)
            
            print(f"Kontrol tamamlandı. {LIVE_BOT_SCAN_INTERVAL_MINUTES} dakika uykuya geçiliyor...")
            time.sleep(sleep_duration_seconds)
        except KeyboardInterrupt:
            print("\nBot manuel olarak durduruldu. Hoşçakalın!")
            send_telegram_notification("🤖 *Cognitive Trade Agent (Canlı Test) Durduruldu!*")
            break
        except Exception as e:
            print(f"HATA: Ana döngüde bir sorun oluştu: {e}")
            send_telegram_notification(f"🤖 BOT HATA VERDİ: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_live_bot()