import os
import json
import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from binance.client import Client as BinanceClient
from strategy_engine import check_live_signal, fetch_data_from_db

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_notification(message: str):
    """Verilen mesajı Telegram botu aracılığıyla gönderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\nUyarı: Telegram token veya chat ID bulunamadı.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4096
    chunks = [message[i:i + max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': chunk, 'parse_mode': 'Markdown'}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print("\nTelegram bildirimi başarıyla gönderildi.")
            else:
                print(f"\nTelegram bildirimi gönderilemedi. Hata: {response.text}")
        except Exception as e:
            print(f"\nTelegram'a bağlanırken bir hata oluştu: {e}")

def analyze_portfolio_opportunities():
    """Tüm varlıklar için en iyi stratejileri bulur ve anlık sinyalleri raporlar."""
    print("Portföy Fırsat Analizi Başlatıldı...")
    try:
        response = supabase.table('backtest_results').select('*, strategies(name, parameters, type)').execute()
        if not response.data:
            print("Analiz edilecek veri bulunamadı."); return

        df = pd.DataFrame(response.data)
        
        # --- AJANIN YENİ BİLGE KURALI ---
        # Sadece en az 1 tane tamamlanmış işlemi olan stratejileri dikkate al.
        df_valid_trades = df[df['total_trades'] > 0]

        if df_valid_trades.empty:
            print("Hiçbir strateji test periyodu boyunca işlem tamamlayamadı."); return

        # Her bir sembol-zaman aralığı kombinasyonu için en iyi stratejiyi bul
        best_strategies = df_valid_trades.loc[df_valid_trades.groupby(['symbol', 'timeframe'])['net_profit_percent'].idxmax()]

        print("\n--- PORTFÖY ANALİZ RAPORU (GÜVENİLİR SONUÇLAR) ---")
        full_report_message = "👑 *Cognitive Agent (Güvenilir) Portföy Raporu* 👑\n\n"

        for index, strategy_record in best_strategies.iterrows():
            symbol, timeframe = strategy_record['symbol'], strategy_record['timeframe']
            strategy_info = {
                'name': strategy_record['strategies']['name'],
                'type': strategy_record['strategies']['type'],
                'params': json.loads(strategy_record['strategies']['parameters'])
            }
            report_line = (f"*{symbol} ({timeframe})*\n"
                           f"En İyi Strateji: `{strategy_info['name']}`\n"
                           f"Net Kâr (Backtest): *{strategy_record['net_profit_percent']:.2f}%* | Maks. Düşüş: {strategy_record['max_drawdown']:.2f}%\n"
                           f"İşlem Sayısı: {strategy_record['total_trades']}\n")
            
            live_data_df = fetch_data_from_db(supabase, symbol, timeframe)
            if not live_data_df.empty:
                live_signal = check_live_signal(live_data_df, strategy_info)
                report_line += f"Canlı Sinyal: `{live_signal}`\n"
            
            print(report_line)
            full_report_message += report_line + "------------------------------------\n"

        send_telegram_notification(full_report_message)

    except Exception as e:
        print(f"Portföy analizi sırasında bir hata oluştu: {e}")

if __name__ == "__main__":
    analyze_portfolio_opportunities()