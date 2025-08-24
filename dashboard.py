import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from supabase import create_client, Client
import os
import json
from datetime import datetime, timedelta

from config import TOTAL_DAYS, TRAINING_DAYS, INITIAL_CAPITAL, CAPITAL_PER_TRADE_PERCENT, MAX_CONCURRENT_POSITIONS
from data_collector import fetch_data_from_supabase
from strategy_engine import generate_equity_curve, calculate_strategy_indicators, check_live_signal

st.set_page_config(page_title="Cognitive Trade Agent", page_icon="🤖", layout="wide")
load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

@st.cache_data(ttl=600)
def load_backtest_data():
    print("Backtest sonuçları Supabase'den çekiliyor...")
    all_data = []
    offset, limit = 0, 1000
    while True:
        try:
            response = supabase.table('backtest_results').select('*, strategies(name, parameters, type)').range(offset, offset + limit - 1).execute()
            if not response.data: break
            all_data.extend(response.data)
            if len(response.data) < limit: break
            offset += limit
        except Exception as e: print(f"Backtest verisi çekilirken hata: {e}"); break
    if not all_data: return pd.DataFrame()
    print(f"Toplam {len(all_data)} adet backtest sonucu başarıyla çekildi.")
    df = pd.DataFrame(all_data)
    df['strategy_name'] = df['strategies'].apply(lambda x: x['name'] if isinstance(x, dict) else 'N/A')
    df['in_sample_total_trades'].fillna(0, inplace=True); df['out_of_sample_total_trades'].fillna(0, inplace=True)
    df['in_sample_profit_percent'].fillna(0, inplace=True); df['out_of_sample_profit_percent'].fillna(0, inplace=True)
    df['out_of_sample_max_drawdown'].fillna(999, inplace=True)
    def calculate_calmar(row):
        profit, drawdown = row['out_of_sample_profit_percent'], row['out_of_sample_max_drawdown']
        if drawdown is not None and drawdown > 0 and drawdown != 999: return profit / drawdown
        elif profit is not None and profit > 0: return 999
        else: return 0
    df['calmar_ratio'] = df.apply(calculate_calmar, axis=1)
    df['calmar_ratio'].fillna(0, inplace=True)
    return df

@st.cache_data(ttl=600)
def load_market_data_for_symbols(_symbols_and_timeframes: tuple):
    all_data = {}
    for item in _symbols_and_timeframes:
        symbol, timeframe = item[0], item[1]
        key = f"{symbol}-{timeframe}"
        all_data[key] = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
    return all_data

@st.cache_data(ttl=600)
def run_portfolio_simulation(_backtest_df, _market_data_dict, selected_keys, min_in_trades, min_out_trades):
    champion_strategies = {}
    filtered_backtest_df = _backtest_df[(_backtest_df['in_sample_total_trades'] >= min_in_trades) & (_backtest_df['out_of_sample_total_trades'] >= min_out_trades)]
    for key in selected_keys:
        symbol, timeframe = key.split('-')
        df_filtered_env = filtered_backtest_df[(filtered_backtest_df['symbol'] == symbol) & (filtered_backtest_df['timeframe'] == timeframe)]
        if not df_filtered_env.empty:
            best_for_env = df_filtered_env.loc[df_filtered_env['out_of_sample_profit_percent'].idxmax()]
            champion_strategies[key] = {'name': best_for_env['strategy_name'], 'type': best_for_env['strategies']['type'],
                'params': json.loads(best_for_env['strategies']['parameters']) if isinstance(best_for_env['strategies']['parameters'], str) else best_for_env['strategies']['parameters']}
    if not champion_strategies:
        st.warning("Belirttiğiniz filtre kriterlerini sağlayan 'şampiyon' strateji bulunamadı."); return None
    validation_data, rich_validation_data = {}, {}
    for key, champ_info in champion_strategies.items():
        df = _market_data_dict.get(key)
        if df is not None and not df.empty:
            split_date = df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
            validation_df = df[df['timestamp'] >= split_date]
            rich_validation_data[key] = calculate_strategy_indicators(validation_df, champ_info)
    if not rich_validation_data: return None
    combined_validation_df = pd.concat(rich_validation_data.values()).sort_values('timestamp').drop_duplicates('timestamp')
    portfolio, trade_log = {'cash': INITIAL_CAPITAL, 'positions': {}}, []
    for timestamp in combined_validation_df['timestamp']:
        current_positions = list(portfolio['positions'].keys())
        for key, rich_df in rich_validation_data.items():
            if key not in champion_strategies: continue
            market_snapshot = rich_df[rich_df['timestamp'] <= timestamp]
            if len(market_snapshot) < 100: continue
            champion_strategy, current_price = champion_strategies[key], market_snapshot.iloc[-1]['close']
            signal = check_live_signal(market_snapshot, champion_strategy)
            if key in current_positions and signal == 'SELL':
                position = portfolio['positions'][key]; sell_value = position['size'] * current_price
                pnl = sell_value - (position['size'] * position['entry_price'])
                portfolio['cash'] += sell_value; trade_log.append({'timestamp': timestamp, 'symbol': key, 'type': 'SELL', 'price': current_price, 'pnl': pnl})
                del portfolio['positions'][key]
            elif key not in current_positions and signal == 'BUY' and len(portfolio['positions']) < MAX_CONCURRENT_POSITIONS:
                capital_to_use = INITIAL_CAPITAL * CAPITAL_PER_TRADE_PERCENT
                if portfolio['cash'] >= capital_to_use:
                    size = capital_to_use / current_price
                    portfolio['positions'][key] = {'size': size, 'entry_price': current_price}
                    portfolio['cash'] -= capital_to_use
                    trade_log.append({'timestamp': timestamp, 'symbol': key, 'type': 'BUY', 'price': current_price, 'pnl': 0})
    final_portfolio_value = portfolio['cash']
    for key, position in portfolio['positions'].items():
        if key in _market_data_dict and not _market_data_dict[key].empty:
            final_portfolio_value += position['size'] * _market_data_dict[key].iloc[-1]['close']
    net_profit = ((final_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    sells = [t for t in trade_log if t['type'] == 'SELL']; total_trades = len(sells)
    wins = len([t for t in sells if t['pnl'] > 0]); win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    return {'final_value': final_portfolio_value, 'net_profit': net_profit, 'win_rate': win_rate, 'total_trades': total_trades, 'trade_log': trade_log, 'champions': champion_strategies}

st.title("🤖 Cognitive Trade Agent - Kontrol Paneli")
backtest_df_raw = load_backtest_data()
if backtest_df_raw.empty:
    st.warning("Veritabanında backtest sonucu bulunamadı. Lütfen önce `orchestrator.py`'yi çalıştırın.")
else:
    st.sidebar.header("🔬 Genel Analiz Filtreleri")
    min_trades_in_sample = st.sidebar.slider("Min. Eğitim İşlemi:", 0, 100, 5)
    min_trades_out_of_sample = st.sidebar.slider("Min. Test İşlemi:", 0, 100, 1)
    max_drawdown_allowed = st.sidebar.slider("Maks. Tolere Edilebilir Düşüş (%):", 0, 100, 50)
    min_calmar_ratio = st.sidebar.slider("Min. Calmar Oranı:", 0.0, 5.0, 0.0, step=0.1)
    backtest_df_filtered = backtest_df_raw[(backtest_df_raw['in_sample_total_trades'] >= min_trades_in_sample) & (backtest_df_raw['out_of_sample_total_trades'] >= min_trades_out_of_sample) & (backtest_df_raw['out_of_sample_max_drawdown'] <= max_drawdown_allowed) & (backtest_df_raw['calmar_ratio'] >= min_calmar_ratio)].copy()
    
    tab1, tab2, tab3 = st.tabs(["Strateji Analizi", "Portföy Simülasyonu", "Büyüme Simülasyonu"])
    with tab1:
        st.header("Strateji Keşif ve Performans Analizi")
        all_symbols_tab1 = sorted(backtest_df_filtered['symbol'].unique())
        if all_symbols_tab1:
            selected_symbol_tab1 = st.sidebar.selectbox("Varlık Seçin:", options=all_symbols_tab1)
            filtered_df_tab1 = backtest_df_filtered[backtest_df_filtered['symbol'] == selected_symbol_tab1]
            sorted_df_tab1 = filtered_df_tab1.sort_values(by='out_of_sample_profit_percent', ascending=False).reset_index(drop=True)
            if not sorted_df_tab1.empty:
                st.subheader(f"🏆 {selected_symbol_tab1} İçin Filtrelenmiş En İyi Stratejiler")
                display_cols = ['strategy_name', 'out_of_sample_profit_percent', 'calmar_ratio', 'out_of_sample_max_drawdown', 'out_of_sample_win_rate', 'out_of_sample_total_trades', 'in_sample_profit_percent', 'in_sample_total_trades']
                st.dataframe(sorted_df_tab1[[col for col in display_cols if col in sorted_df_tab1.columns]])
                st.subheader("📊 En İyi 10 Stratejinin Performans Grafiği")
                top_10_df = sorted_df_tab1.head(10)
                fig = px.bar(top_10_df, x='strategy_name', y='out_of_sample_profit_percent', title=f"{selected_symbol_tab1} İçin En Güvenilir Stratejiler", labels={'strategy_name': 'Strateji Adı', 'out_of_sample_profit_percent': 'Gerçek Test Kârı (%)'}, color='calmar_ratio', color_continuous_scale=px.colors.sequential.Tealgrn, hover_data=['in_sample_profit_percent', 'out_of_sample_total_trades', 'out_of_sample_win_rate'])
                fig.update_xaxes(tickangle=45); st.plotly_chart(fig, use_container_width=True)
            else: st.info(f"{selected_symbol_tab1} için belirlediğiniz filtrelerde geçerli bir strateji bulunamadı.")
        else: st.info("Belirlediğiniz filtrelerde görüntülenecek hiçbir strateji bulunamadı.")
    with tab2:
        st.header("Geçmişe Dönük Portföy Yönetimi Simülasyonu")
        st.markdown("Seçtiğiniz varlıklar için ajan tarafından **filtreleri sağlayan** en iyi stratejilerle bir portföy yönetilseydi ne olurdu?")
        all_environments = sorted(backtest_df_filtered.drop_duplicates(subset=['symbol', 'timeframe'])[['symbol', 'timeframe']].apply(lambda x: f"{x[0]}-{x[1]}", axis=1).tolist())
        if not all_environments:
            st.warning("Portföy simülasyonu için, belirlediğiniz filtreleri sağlayan hiçbir varlık bulunamadı.")
        else:
            selected_keys = st.multiselect("Portföyünüz için ortamları seçin:", options=all_environments, default=all_environments[:len(all_environments) if len(all_environments) <= 4 else 4])
            if st.button("Portföy Simülasyonunu Çalıştır"):
                if not selected_keys: st.error("Lütfen simülasyon için en az bir ortam seçin.")
                else:
                    with st.spinner("Piyasa verileri yükleniyor ve simülasyon çalıştırılıyor..."):
                        symbols_and_timeframes_to_load = tuple(set((key.split('-')[0], key.split('-')[1]) for key in selected_keys))
                        market_data = load_market_data_for_symbols(symbols_and_timeframes_to_load)
                        portfolio_results = run_portfolio_simulation(backtest_df_filtered, market_data, selected_keys, min_trades_in_sample, min_trades_out_of_sample)
                    st.subheader("📈 Portföy Simülasyonu Nihai Sonucu")
                    if portfolio_results:
                        st.write("**Simülasyonda Kullanılan Şampiyon Stratejiler:**"); st.json({k: v['name'] for k, v in portfolio_results['champions'].items()})
                        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                        p_col1.metric("Bitiş Değeri", f"${portfolio_results['final_value']:,.2f}"); p_col2.metric("Net Portföy Kârı", f"{portfolio_results['net_profit']:.2f}%"); p_col3.metric("Genel Kazanma Oranı", f"{portfolio_results['win_rate']:.2f}%"); p_col4.metric("Toplam İşlem", portfolio_results['total_trades'])
                        st.subheader("İşlem Günlüğü"); st.dataframe(pd.DataFrame(portfolio_results['trade_log']))
                    else: st.error("Simülasyon sonucu üretilemedi.")
    with tab3:
        st.header("Strateji Büyüme Eğrisi Simülasyonu")
        st.markdown("Filtrelediğiniz sonuçlar arasından seçtiğiniz bir stratejinin, belirlediğiniz sermaye ile geçmiş **test periyodunda** nasıl bir performans serüveni izleyeceğini görün.")
        if backtest_df_filtered.empty:
            st.warning("Simülasyon için, filtrelerinize uyan geçerli bir strateji bulunamadı.")
        else:
            col1, col2 = st.columns(2)
            start_capital_input = col1.number_input("Başlangıç Sermayesi (USDT):", min_value=100.0, value=10000.0, step=1000.0)
            strategy_options = backtest_df_filtered.apply(lambda row: f"{row['strategy_name']} ({row['symbol']}-{row['timeframe']})", axis=1).unique()
            selected_strategy_str = col2.selectbox("Simüle edilecek stratejiyi seçin:", options=strategy_options)
            if st.button("Büyüme Grafiğini Oluştur"):
                if not selected_strategy_str: st.warning("Lütfen bir strateji seçin."); st.stop()
                strategy_name_to_find, env_key = selected_strategy_str.split(' (')[0], selected_strategy_str.split('(')[1].replace(')', '')
                symbol, timeframe = env_key.split('-')
                strategy_details_row = backtest_df_raw[(backtest_df_raw['strategy_name'] == strategy_name_to_find) & (backtest_df_raw['symbol'] == symbol) & (backtest_df_raw['timeframe'] == timeframe)].iloc[0]
                strategy_info = {'name': strategy_details_row['strategies']['name'], 'type': strategy_details_row['strategies']['type'], 'params': json.loads(strategy_details_row['strategies']['parameters']) if isinstance(strategy_details_row['strategies']['parameters'], str) else strategy_details_row['strategies']['parameters']}
                with st.spinner("Büyüme eğrisi oluşturuluyor..."):
                    market_data_df = fetch_data_from_supabase(symbol, timeframe, TOTAL_DAYS)
                    split_date = market_data_df['timestamp'].min() + timedelta(days=TRAINING_DAYS)
                    validation_df = market_data_df[market_data_df['timestamp'] >= split_date]
                    equity_df = generate_equity_curve(validation_df, strategy_info, start_capital_input)
                if not equity_df.empty:
                    st.subheader(f"📈 '{strategy_name_to_find}' Stratejisinin Büyüme Eğrisi")
                    final_value, profit_percent = equity_df['portfolio_value'].iloc[-1], ((equity_df['portfolio_value'].iloc[-1] / start_capital_input) - 1) * 100
                    st.metric("Bitiş Değeri", f"${final_value:,.2f}", f"{profit_percent:.2f}%")
                    fig = px.line(equity_df, x='timestamp', y='portfolio_value', title=f"Sermaye Büyüme Grafiği", labels={'timestamp': 'Tarih', 'portfolio_value': 'Portföy Değeri (USDT)'})
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Büyüme eğrisi oluşturulamadı.")