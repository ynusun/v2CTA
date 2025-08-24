import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta
import json
import time

from config import (
    MARKET_ENVIRONMENTS, TOTAL_DAYS, TRAINING_DAYS,
    INITIAL_CAPITAL, CAPITAL_PER_TRADE_PERCENT, MAX_CONCURRENT_POSITIONS, USE_STOP_LOSS, ATR_PERIOD, ATR_MULTIPLIER
)
from strategy_engine import run_backtest, check_live_signal, calculate_strategy_indicators
from orchestrator import generate_strategies, get_or_create_strategy_in_db
from data_collector import fetch_data_from_supabase, fetch_and_store_all_klines

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def run_portfolio_simulation():
    CHAMPIONS_FILE = 'champions.json'
    champion_strategies = {}
    
    if os.path.exists(CHAMPIONS_FILE):
        print(f"'{CHAMPIONS_FILE}' bulundu, Eğitim Fazı atlanıyor...")
        with open(CHAMPIONS_FILE, 'r') as f:
            champion_strategies = json.load(f)

    print("\n" + "="*60 + "\nFAZ 1: VERİ TOPLAMA VE HAZIRLIK\n" + "="*60)
    all_asset_data, training_data, validation_data = {}, {}, {}
    for market in MARKET_ENVIRONMENTS:
        symbol, timeframe = market['symbol'], market['timeframe']
        key = f"{symbol}-{timeframe}"
        df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
        if df.empty:
             print(f"{symbol}-{timeframe} için veri bulunamadı, Binance'ten çekiliyor...")
             start_date_str = (datetime.now() - timedelta(days=TOTAL_DAYS)).strftime("%d %b, %Y")
             fetch_and_store_all_klines(symbol, timeframe, start_date_str)
             df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
        all_asset_data[key] = df
        if not df.empty:
            split_date = df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
            training_data[key] = df[df['timestamp'] < split_date]
            validation_data[key] = df[df['timestamp'] >= split_date]

    if not champion_strategies:
        print("\n" + "="*60 + "\nFAZ 2: EĞİTİM (HER ORTAM İÇİN EN İYİ STRATEJİ BULUNUYOR)\n" + "="*60)
        strategies_to_test = generate_strategies()
        for key, train_df in training_data.items():
            if train_df.empty: continue
            print(f"\n--- {key} için en iyi strateji aranıyor... ---")
            best_strategy_info, best_performance = None, -100
            for strategy_info in strategies_to_test:
                results = run_backtest(train_df.copy(), strategy_info=strategy_info)
                if results and results['net_profit_percent'] > best_performance:
                    best_performance, best_strategy_info = results['net_profit_percent'], strategy_info
            if best_strategy_info:
                champion_strategies[key] = best_strategy_info
                print(f"==> {key} için Şampiyon Strateji: {best_strategy_info['name']} ({best_performance:.2f}%)")
        with open(CHAMPIONS_FILE, 'w') as f:
            json.dump(champion_strategies, f, indent=4)
        print(f"\nŞampiyon stratejiler '{CHAMPIONS_FILE}' dosyasına kaydedildi.")

    print("\n" + "="*60 + "\nFAZ 3: PORTFÖY SİMÜLASYONU (GÖRÜLMEMİŞ VERİ ÜZERİNDE)\n" + "="*60)
    
    print("Simülasyon için tüm test verisi göstergeleri hesaplanıyor...")
    rich_validation_data = {}
    for key, df in validation_data.items():
        if not df.empty and key in champion_strategies:
            rich_validation_data[key] = calculate_strategy_indicators(df, champion_strategies[key])
    print("Gösterge hesaplaması tamamlandı. Simülasyon başlıyor.")

    valid_validation_data = {k: v for k, v in rich_validation_data.items() if not v.empty}
    if not valid_validation_data:
        print("Test için kullanılabilir veri yok."); return
    combined_validation_df = pd.concat(valid_validation_data.values()).sort_values('timestamp').drop_duplicates('timestamp')
    portfolio, trade_log = {'cash': INITIAL_CAPITAL, 'positions': {}}, []
    atr_col_name = f"ATRr_{ATR_PERIOD}"

    for timestamp in combined_validation_df['timestamp']:
        current_positions = list(portfolio['positions'].keys())
        if USE_STOP_LOSS:
            for pos_key in current_positions:
                market_snapshot = rich_validation_data[pos_key][rich_validation_data[pos_key]['timestamp'] <= timestamp]
                if market_snapshot.empty: continue
                current_low_price, position = market_snapshot.iloc[-1]['low'], portfolio['positions'][pos_key]
                if current_low_price <= position['stop_loss']:
                    sell_price = position['stop_loss']
                    sell_value = position['size'] * sell_price
                    pnl = sell_value - (position['size'] * position['entry_price'])
                    portfolio['cash'] += sell_value
                    trade_log.append({'symbol': pos_key, 'type': 'STOP_LOSS', 'price': sell_price, 'pnl': pnl})
                    print(f"{timestamp} | STOP-LOSS: {pos_key} | Fiyat: {sell_price:.2f} | PNL: ${pnl:.2f} | Kasa: ${portfolio['cash']:.2f}")
                    del portfolio['positions'][pos_key]

        current_positions = list(portfolio['positions'].keys())
        for key, rich_df in rich_validation_data.items():
            if key not in champion_strategies: continue
            market_snapshot = rich_df[rich_df['timestamp'] <= timestamp]
            if len(market_snapshot) < 100: continue
            champion_strategy, current_price = champion_strategies[key], market_snapshot.iloc[-1]['close']
            signal = check_live_signal(market_snapshot, champion_strategy)
            if key in current_positions and signal == 'SELL':
                position = portfolio['positions'][key]
                sell_value, pnl = position['size'] * current_price, (position['size'] * current_price) - (position['size'] * position['entry_price'])
                portfolio['cash'] += sell_value
                trade_log.append({'symbol': key, 'type': 'SELL', 'price': current_price, 'pnl': pnl})
                print(f"{timestamp} | SAT: {key} | Fiyat: {current_price:.2f} | PNL: ${pnl:.2f} | Kasa: ${portfolio['cash']:.2f}")
                del portfolio['positions'][key]
            elif key not in current_positions and signal == 'BUY' and len(portfolio['positions']) < MAX_CONCURRENT_POSITIONS:
                capital_to_use = INITIAL_CAPITAL * CAPITAL_PER_TRADE_PERCENT
                if portfolio['cash'] >= capital_to_use:
                    size = capital_to_use / current_price
                    stop_loss_price = 0.0
                    if USE_STOP_LOSS and atr_col_name in market_snapshot.columns and pd.notna(market_snapshot.iloc[-1][atr_col_name]):
                        atr_value = market_snapshot.iloc[-1][atr_col_name]
                        stop_loss_price = current_price - (atr_value * ATR_MULTIPLIER)
                    portfolio['positions'][key] = {'size': size, 'entry_price': current_price, 'stop_loss': stop_loss_price}
                    portfolio['cash'] -= capital_to_use
                    trade_log.append({'symbol': key, 'type': 'BUY', 'price': current_price, 'pnl': 0})
                    print(f"{timestamp} | AL: {key} | Fiyat: {current_price:.2f} | Büyüklük: {size:.4f} | Kasa: ${portfolio['cash']:.2f}")

    print("\n" + "*"*60 + "\n                 PORTFÖY SİMÜLASYONU NİHAİ SONUCU\n" + "*"*60)
    final_portfolio_value = portfolio['cash']
    for key, position in portfolio['positions'].items():
        if key in all_asset_data and not all_asset_data[key].empty:
            final_portfolio_value += position['size'] * all_asset_data[key].iloc[-1]['close']
    net_profit_percent = ((final_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    total_trades = len([t for t in trade_log if t['type'] in ['SELL', 'STOP_LOSS']])
    winning_trades = len([t for t in trade_log if t['pnl'] > 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    print("Test Edilen Ortamlar:", list(champion_strategies.keys()))
    print(f"Başlangıç Sermayesi: ${INITIAL_CAPITAL:,.2f}"); print(f"Bitiş Portföy Değeri: ${final_portfolio_value:,.2f}")
    print(f"\n--> Net Portföy Kâr/Zararı: {net_profit_percent:.2f}%"); print(f"--> Kazanma Oranı (Genel): {win_rate:.2f}%"); print(f"--> Toplam Kapanan İşlem: {total_trades}"); print("*"*60)

if __name__ == "__main__":
    run_portfolio_simulation()