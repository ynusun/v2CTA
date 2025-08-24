import os
import json
import itertools
import importlib
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta
import httpx

from strategy_engine import run_backtest, detect_market_regime
from data_collector import fetch_data_from_supabase
from config import ACTIVE_INDICATORS, MARKET_ENVIRONMENTS, TOTAL_DAYS, TRAINING_DAYS

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def get_or_create_strategy_in_db(strategy_info: dict) -> str:
    global supabase
    name = strategy_info['name']
    params_json = json.dumps(strategy_info['params'], sort_keys=True)
    try:
        response = supabase.table('strategies').select('id').eq('parameters', params_json).eq('type', strategy_info['type']).execute()
    except httpx.RemoteProtocolError:
        print("Bağlantı hatası (get_or_create), yeniden bağlanılıyor...")
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        response = supabase.table('strategies').select('id').eq('parameters', params_json).eq('type', strategy_info['type']).execute()
    if response.data: return response.data[0]['id']
    else:
        print(f"Yeni strateji oluşturuluyor: '{name}'")
        insert_data = {'name': name, 'description': 'Ajan tarafından dinamik olarak sentezlendi.', 'parameters': params_json, 'type': strategy_info['type']}
        response = supabase.table('strategies').insert(insert_data).execute()
        if response.data: return response.data[0]['id']
        else: raise Exception(f"Yeni strateji eklenemedi: {response}")

def generate_strategies(regime: str):
    print(f"Strateji Sentezleyici Başlatıldı: '{regime}' rejimine uygun kurallar aranıyor...")
    base_rules = {'buy': [], 'sell': []}
    for indicator_name, is_active in ACTIVE_INDICATORS.items():
        if is_active:
            try:
                plugin_module = importlib.import_module(f"indicators.{indicator_name}_plugin")
                all_rules_from_plugin = plugin_module.get_rules()
                filtered_buy_rules = [r for r in all_rules_from_plugin.get('buy', []) if regime in r['rule'].get('regimes', [])]
                filtered_sell_rules = [r for r in all_rules_from_plugin.get('sell', []) if regime in r['rule'].get('regimes', [])]
                if filtered_buy_rules:
                    base_rules['buy'].extend(filtered_buy_rules)
                    base_rules['sell'].extend(filtered_sell_rules)
                    print(f"-> {indicator_name.upper()} plugin'inden {len(filtered_buy_rules)} adet uygun kural yüklendi.")
            except Exception as e: print(f"UYARI: {indicator_name}_plugin ile ilgili bir sorun var: {e}")
    print(f"Kural havuzu tamamlandı. {len(base_rules['buy'])} adet temel alım kuralı üretildi.")
    strategies = []
    # 1'li stratejiler
    for rule_info in base_rules['buy']:
        sell_rule = next((r['rule'] for r in base_rules['sell'] if r['name'] == rule_info['name'].replace('_Buy', '_Sell')), None)
        if sell_rule: strategies.append({'name': rule_info['name'], 'type': 'CONFLUENCE','params': {'buy_rules': [rule_info['rule']], 'sell_rules': [sell_rule], 'buy_quorum': 1, 'sell_quorum': 1}})
    # 2'li stratejiler
    if len(base_rules['buy']) >= 2:
        for buy_combo in itertools.combinations(base_rules['buy'], 2):
            try:
                sell_combo_rules = [next(r['rule'] for r in base_rules['sell'] if r['name'] == rule['name'].replace('_Buy', '_Sell')) for rule in buy_combo]
                if len(sell_combo_rules) == 2: strategies.append({'name': f"Sentez_{buy_combo[0]['name']}_&_{buy_combo[1]['name']}", 'type': 'CONFLUENCE','params': {'buy_rules': [c['rule'] for c in buy_combo], 'sell_rules': sell_combo_rules, 'buy_quorum': 2, 'sell_quorum': 2}})
            except StopIteration: continue
    print(f"Toplam {len(strategies)} adet strateji (tekli ve sentezlenmiş) test için hazırlandı.")
    return strategies

def main():
    global supabase
    for env in MARKET_ENVIRONMENTS:
        symbol, timeframe = env['symbol'], env['timeframe']
        print("\n" + "="*60); print(f"PİYASA ANALİZİ BAŞLATILIYOR: {symbol} - {timeframe}"); print("="*60)
        full_df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
        if full_df.empty: continue
        split_date = full_df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
        training_df, validation_df = full_df[full_df['timestamp'] < split_date], full_df[full_df['timestamp'] >= split_date]
        if training_df.empty or validation_df.empty:
            print(f"{symbol}-{timeframe} için veri, eğitim/test setlerine bölünemedi, atlanıyor."); continue
        
        current_regime = detect_market_regime(training_df)
        print(f"Eğitim verisi için piyasa rejimi tespit edildi: {current_regime}")
        strategies_to_test = generate_strategies(current_regime)
        total_strategies = len(strategies_to_test)
        if total_strategies == 0:
            print("Bu rejim için test edilecek uygun strateji bulunamadı."); continue
        
        for i, strategy_info in enumerate(strategies_to_test):
            print("-" * 50); print(f"Strateji {i+1}/{total_strategies} analiz ediliyor: {strategy_info['name']}")
            try:
                strategy_id = get_or_create_strategy_in_db(strategy_info)
                response = supabase.table('backtest_results').select('strategy_id').eq('strategy_id', strategy_id).eq('symbol', symbol).eq('timeframe', timeframe).execute()
            except httpx.RemoteProtocolError as e:
                print(f"UYARI: Supabase bağlantısı (okuma sırasında) zaman aşımına uğradı. Yeniden bağlanılıyor...")
                supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
                strategy_id = get_or_create_strategy_in_db(strategy_info)
                response = supabase.table('backtest_results').select('strategy_id').eq('strategy_id', strategy_id).eq('symbol', symbol).eq('timeframe', timeframe).execute()
            if response.data:
                print("Bu strateji bu piyasada daha önce test edilmiş, atlanıyor."); continue
            
            in_sample_results = run_backtest(training_df.copy(), strategy_info=strategy_info)
            out_of_sample_results = run_backtest(validation_df.copy(), strategy_info=strategy_info)
            if in_sample_results and out_of_sample_results:
                db_record = {'symbol': symbol, 'timeframe': timeframe, 'strategy_id': strategy_id, 'start_date': full_df['timestamp'].min().isoformat(), 'end_date': full_df['timestamp'].max().isoformat(), 'in_sample_profit_percent': in_sample_results['net_profit_percent'], 'in_sample_win_rate': in_sample_results['win_rate'], 'in_sample_max_drawdown': in_sample_results['max_drawdown'], 'in_sample_total_trades': in_sample_results['total_trades'], 'out_of_sample_profit_percent': out_of_sample_results['net_profit_percent'], 'out_of_sample_win_rate': out_of_sample_results['win_rate'], 'out_of_sample_max_drawdown': out_of_sample_results['max_drawdown'], 'out_of_sample_total_trades': out_of_sample_results['total_trades'],}
                try:
                    supabase.table('backtest_results').insert(db_record).execute()
                    print("Walk-Forward analizi sonucu başarıyla kaydedildi.")
                except httpx.RemoteProtocolError as e:
                    print(f"UYARI: Supabase bağlantısı (yazma sırasında) zaman aşımına uğradı. Yeniden bağlanılıyor...")
                    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
                    supabase.table('backtest_results').insert(db_record).execute()
                    print("İşlem yeniden denendi ve başarıyla kaydedildi.")
                except Exception as e:
                    print(f"Veritabanına kayıt sırasında beklenmedik bir hata oluştu: {e}")
    print("\n" + "="*60); print("TÜM PİYASA ANALİZLERİ TAMAMLANDI."); print("="*60)

if __name__ == "__main__":
    main()