import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta

from strategy_engine import run_backtest
from orchestrator import generate_strategies, get_or_create_strategy_in_db

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# --- GÜNCELLENMİŞ FONKSİYON ---
def fetch_full_dataset(symbol, timeframe, days_total=180) -> pd.DataFrame:
    """
    Veritabanından belirtilen periyot için TÜM veriyi parça parça çeker.
    """
    print(f"{days_total} günlük veri '{symbol}-{timeframe}' için Supabase'den çekiliyor...")
    start_date = datetime.now() - timedelta(days=days_total)
    
    all_data = []
    offset = 0
    limit = 1000 # Supabase'in varsayılan limiti

    while True:
        try:
            # Veriyi parça parça çekmek için .range() kullanıyoruz
            response = supabase.table('market_data').select('*') \
                .eq('symbol', symbol) \
                .eq('timeframe', timeframe) \
                .gte('timestamp', start_date.isoformat()) \
                .order('timestamp', desc=False) \
                .range(offset, offset + limit - 1) \
                .execute()

            if not response.data:
                # Eğer daha fazla veri gelmiyorsa döngüyü kır
                break
            
            all_data.extend(response.data)
            offset += limit
            
        except Exception as e:
            print(f"Veri çekme sırasında hata: {e}")
            break
            
    if not all_data: return pd.DataFrame()
    
    print(f"Toplam {len(all_data)} adet veri satırı başarıyla çekildi.")
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def main():
    """İleriye yönelik analiz simülasyonunu çalıştırır."""
    
    TARGET_SYMBOL = "BTCUSDT"
    TARGET_TIMEFRAME = "1h"
    TOTAL_DAYS = 180
    TRAINING_DAYS = 90

    full_df = fetch_full_dataset(TARGET_SYMBOL, TARGET_TIMEFRAME, TOTAL_DAYS)
    if full_df.empty:
        print("Analiz için yeterli veri bulunamadı."); return

    split_date = full_df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
    training_df = full_df[full_df['timestamp'] < split_date]
    validation_df = full_df[full_df['timestamp'] >= split_date]

    if validation_df.empty:
        print("HATA: Test seti oluşturulamadı. Veri periyodunu veya data_collector.py'yi kontrol edin.")
        return

    print(f"\nVeri seti hazırlandı:")
    print(f"- Eğitim Seti: {training_df.iloc[0]['timestamp'].date()} -> {training_df.iloc[-1]['timestamp'].date()} ({len(training_df)} mum)")
    print(f"- Test Seti:    {validation_df.iloc[0]['timestamp'].date()} -> {validation_df.iloc[-1]['timestamp'].date()} ({len(validation_df)} mum)")

    print("\n" + "="*60); print("FAZ 1: EĞİTİM - En iyi strateji ilk 3 aylık veride aranıyor..."); print("="*60)
    
    strategies_to_test = generate_strategies()
    best_strategy_info = None
    best_performance = -100

    for strategy_info in strategies_to_test:
        print("-" * 50)
        strategy_id = get_or_create_strategy_in_db(strategy_info)
        backtest_results = run_backtest(training_df.copy(), strategy_info=strategy_info)
        if backtest_results and backtest_results['net_profit_percent'] > best_performance:
            best_performance = backtest_results['net_profit_percent']
            best_strategy_info = strategy_info
    
    if not best_strategy_info:
        print("Eğitim fazında kârlı bir strateji bulunamadı."); return

    print("\n" + "="*60); print(f"EĞİTİM TAMAMLANDI! En iyi strateji bulundu: {best_strategy_info['name']}"); print(f"Eğitim Seti Performansı: {best_performance:.2f}%"); print("="*60)

    print("\n" + "="*60); print("FAZ 2: TEST - En iyi strateji, daha önce görmediği son 3 aylık veride deneniyor..."); print("="*60)

    final_results = run_backtest(validation_df.copy(), strategy_info=best_strategy_info)

    print("\n" + "*"*60); print("                 İLERİYE YÖNELİK ANALİZ NİHAİ SONUCU"); print("*"*60)
    print(f"Eğitim setinde en kârlı bulunan strateji: '{best_strategy_info['name']}'")
    print("\nBu stratejinin, daha önce HİÇ GÖRMEDİĞİ son 3 aylık verideki performansı:")
    if final_results:
        print(f"\n--> Net Kâr/Zarar: {final_results['net_profit_percent']:.2f}%")
        print(f"--> Kazanma Oranı: {final_results['win_rate']:.2f}%")
        print(f"--> Maksimum Düşüş: {final_results['max_drawdown']:.2f}%")
        print(f"--> Toplam İşlem: {final_results['total_trades']}")
    else:
        print("\nTest setinde bir sonuç üretilemedi.")
    print("*"*60)

if __name__ == "__main__":
    main()