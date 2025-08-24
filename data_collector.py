import os
from dotenv import load_dotenv
from binance.client import Client
from supabase import create_client, Client as SupabaseClient
from datetime import datetime, timedelta
import time
import pandas as pd
from config import MARKET_ENVIRONMENTS, TOTAL_DAYS

load_dotenv()
supabase: SupabaseClient = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
binance_client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

def fetch_data_from_supabase(symbol: str, timeframe: str, days_total: int) -> pd.DataFrame:
    """Veritabanından belirtilen periyot için TÜM veriyi parça parça çeker."""
    print(f"\n{days_total} günlük veri '{symbol}-{timeframe}' için Supabase'den çekiliyor...")
    start_date = datetime.now() - timedelta(days=days_total)
    
    all_data = []
    offset, limit = 0, 1000
    while True:
        try:
            response = supabase.table('market_data').select('*').eq('symbol', symbol).eq('timeframe', timeframe).gte('timestamp', start_date.isoformat()).order('timestamp', desc=False).range(offset, offset + limit - 1).execute()
            if not response.data: break
            all_data.extend(response.data)
            if len(response.data) < limit: break
            offset += limit
        except Exception as e: 
            print(f"Supabase'den veri çekme sırasında hata: {e}"); break
            
    if not all_data: 
        print(f"'{symbol}-{timeframe}' için Supabase'de veri bulunamadı.")
        return pd.DataFrame()
    
    print(f"Toplam {len(all_data)} adet veri satırı başarıyla çekildi.")
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def fetch_and_store_all_klines(symbol: str, timeframe: str, start_date_str: str):
    """Belirtilen başlangıç tarihinden bugüne kadar olan TÜM k-line verilerini Binance'ten çeker."""
    print(f"'{symbol}' için '{start_date_str}' tarihinden itibaren '{timeframe}' verileri Binance'ten çekiliyor...")
    try:
        klines_generator = binance_client.get_historical_klines_generator(symbol, timeframe, start_str=start_date_str)
        all_klines = list(klines_generator)
        if not all_klines:
            print(f"'{symbol}-{timeframe}' için Binance'te veri bulunamadı."); return
        
        print(f"{len(all_klines)} adet mum verisi bulundu. Supabase'e kaydediliyor...")
        data_to_insert = [{'timestamp': datetime.fromtimestamp(k[0] / 1000).isoformat(), 'symbol': symbol, 'timeframe': timeframe, 'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])} for k in all_klines]
        
        supabase.table('market_data').upsert(data_to_insert, on_conflict='timestamp,symbol,timeframe').execute()
        print(f"'{symbol}' - '{timeframe}' verileri başarıyla kaydedildi/güncellendi!")
    except Exception as e:
        print(f"'{symbol}-{timeframe}' için veri çekilirken bir hata meydana geldi: {e}")

if __name__ == "__main__":
    start_date = datetime.now() - timedelta(days=TOTAL_DAYS)
    start_date_str = start_date.strftime("%d %b, %Y")
    for market in MARKET_ENVIRONMENTS:
        fetch_and_store_all_klines(market['symbol'], market['timeframe'], start_date_str)
        time.sleep(1)