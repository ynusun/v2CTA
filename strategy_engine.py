import pandas as pd
import pandas_ta as ta
from supabase import Client as SupabaseClient
from config import USE_STOP_LOSS, ATR_PERIOD, ATR_MULTIPLIER, REGIME_DETECTION_PERIOD
import numpy as np

def detect_market_regime(df: pd.DataFrame) -> str:
    """Verilen veri setinin genel piyasa rejimini (Boğa, Ayı, Yatay) belirler."""
    if df.empty or len(df) < REGIME_DETECTION_PERIOD:
        return "Unknown"
    df_copy = df.copy()
    slow_ma_col = f"SMA_{REGIME_DETECTION_PERIOD}"
    if slow_ma_col not in df_copy.columns:
        df_copy.ta.sma(length=REGIME_DETECTION_PERIOD, append=True)
    last_price = df_copy['close'].iloc[-1]
    last_ma = df_copy[slow_ma_col].iloc[-1]
    if pd.isna(last_ma):
        return "Unknown"
    atr_col = f"ATRr_{ATR_PERIOD}"
    if atr_col not in df_copy.columns:
        df_copy.ta.atr(length=ATR_PERIOD, append=True)
    atr_ratio = (df_copy[atr_col].iloc[-20:].mean() / df_copy['close'].iloc[-20:].mean()) * 100
    price_ma_ratio = abs((last_price / last_ma) - 1)
    if price_ma_ratio < 0.02 and atr_ratio < 1.0:
        return "Sideways"
    elif last_price > last_ma:
        return "Bull"
    else:
        return "Bear"

def calculate_ott(df: pd.DataFrame, length: int = 2, percent: float = 1.4) -> pd.DataFrame:
    df_copy = df.copy()
    ott_col_name = f"OTT_{length}_{float(percent)}"
    if ott_col_name in df_copy.columns: return df_copy
    mavg_series = ta.vidya(df_copy['close'], length=length).ffill()
    fark = mavg_series * (percent / 100.0)
    long_stop, short_stop = mavg_series - fark, mavg_series + fark
    dir_series = pd.Series(1.0, index=df_copy.index)
    for i in range(1, len(df_copy)):
        dir_prev, long_stop_prev, short_stop_prev = dir_series.iloc[i-1], long_stop.iloc[i-1], short_stop.iloc[i-1]
        if mavg_series.iloc[i] > long_stop_prev: long_stop.iloc[i] = max(long_stop.iloc[i], long_stop_prev)
        if mavg_series.iloc[i] < short_stop_prev: short_stop.iloc[i] = min(short_stop.iloc[i], short_stop_prev)
        if dir_prev == -1 and mavg_series.iloc[i] > short_stop_prev: dir_series.iloc[i] = 1
        elif dir_prev == 1 and mavg_series.iloc[i] < long_stop_prev: dir_series.iloc[i] = -1
        else: dir_series.iloc[i] = dir_prev
    mt_series = long_stop.where(dir_series == 1, short_stop)
    ott_series = mt_series * np.where(mavg_series > mt_series, 1 + percent / 100, 1 - percent / 100)
    df_copy[ott_col_name] = ott_series.shift(2)
    return df_copy

def calculate_strategy_indicators(df: pd.DataFrame, strategy_info: dict) -> pd.DataFrame:
    df_copy = df.copy()
    params = strategy_info.get('params', {})
    rules = params.get('buy_rules', []) + params.get('sell_rules', [])
    if USE_STOP_LOSS and f"ATRr_{ATR_PERIOD}" not in df_copy.columns:
        df_copy.ta.atr(length=ATR_PERIOD, append=True)
    for rule in rules:
        p, indicator = rule.get('params', {}), rule.get('indicator')
        try:
            if indicator == 'RSI' and f"RSI_{p['rsi_period']}" not in df_copy.columns: df_copy.ta.rsi(length=p['rsi_period'], append=True)
            elif indicator == 'MACD' and f"MACD_{p['fast_period']}_{p['slow_period']}_{p['signal_period']}" not in df_copy.columns: df_copy.ta.macd(fast=p['fast_period'], slow=p['slow_period'], signal=p['signal_period'], append=True)
            elif indicator == 'MA_CROSS':
                if f"SMA_{p['fast_period']}" not in df_copy.columns: df_copy.ta.sma(length=p['fast_period'], append=True)
                if f"SMA_{p['slow_period']}" not in df_copy.columns: df_copy.ta.sma(length=p['slow_period'], append=True)
            elif indicator == 'BBANDS' and f"BBL_{p['length']}_{float(p['std'])}" not in df_copy.columns: df_copy.ta.bbands(length=p['length'], std=p['std'], append=True)
            elif indicator == 'OBV':
                if 'OBV' not in df_copy.columns: df_copy.ta.obv(append=True)
                if 'obv_sma_period' in p and f"OBV_SMA_{p['obv_sma_period']}" not in df_copy.columns:
                    df_copy.ta.sma(close=df_copy['OBV'], length=p['obv_sma_period'], append=True, col_names=(f"OBV_SMA_{p['obv_sma_period']}",))
            elif indicator == 'OTT': df_copy = calculate_ott(df_copy, length=p['length'], percent=p['percent'])
        except Exception as e: print(f"Gösterge hesaplama hatası: {indicator} - {e}")
    return df_copy

def _evaluate_rule(rule: dict, prev_row: pd.Series, curr_row: pd.Series) -> bool:
    try:
        indicator, condition, params = rule.get('indicator'), rule.get('condition'), rule.get('params', {})
        if indicator == 'RSI':
            col, val = f"RSI_{params['rsi_period']}", rule.get('value')
            if pd.isna(prev_row[col]) or pd.isna(curr_row[col]): return False
            if condition == 'cross_above': return prev_row[col] < val and curr_row[col] >= val
            if condition == 'cross_below': return prev_row[col] > val and curr_row[col] <= val
        elif indicator == 'MACD':
            p = params; macd_col, signal_col = f"MACD_{p['fast_period']}_{p['slow_period']}_{p['signal_period']}", f"MACDs_{p['fast_period']}_{p['slow_period']}_{p['signal_period']}"
            if pd.isna(prev_row[macd_col]) or pd.isna(prev_row[signal_col]): return False
            if condition == 'cross_above_signal': return prev_row[macd_col] < prev_row[signal_col] and curr_row[macd_col] >= curr_row[signal_col]
            if condition == 'cross_below_signal': return prev_row[macd_col] > prev_row[signal_col] and curr_row[macd_col] <= curr_row[signal_col]
        elif indicator == 'MA_CROSS':
            p = params; fast_col, slow_col = f"SMA_{p['fast_period']}", f"SMA_{p['slow_period']}"
            if pd.isna(prev_row[fast_col]) or pd.isna(prev_row[slow_col]): return False
            if condition == 'cross_above': return prev_row[fast_col] < prev_row[slow_col] and curr_row[fast_col] >= curr_row[slow_col]
            if condition == 'cross_below': return prev_row[fast_col] > prev_row[slow_col] and curr_row[fast_col] <= curr_row[slow_col]
        elif indicator == 'BBANDS':
            p = params; lower_band_col, upper_band_col = f"BBL_{p['length']}_{float(p['std'])}", f"BBU_{p['length']}_{float(p['std'])}"
            if pd.isna(prev_row[lower_band_col]) or pd.isna(curr_row[lower_band_col]): return False
            if condition == 'price_cross_above_lower': return prev_row['close'] < prev_row[lower_band_col] and curr_row['close'] >= curr_row[lower_band_col]
            if condition == 'price_cross_below_upper': return prev_row['close'] > prev_row[upper_band_col] and curr_row['close'] <= curr_row[upper_band_col]
        elif indicator == 'OBV':
            p = params; obv_col, obv_sma_col = 'OBV', f"OBV_SMA_{p['obv_sma_period']}"
            if pd.isna(prev_row[obv_col]) or pd.isna(prev_row[obv_sma_col]): return False
            if condition == 'cross_above_sma': return prev_row[obv_col] < prev_row[obv_sma_col] and curr_row[obv_col] >= curr_row[obv_sma_col]
            if condition == 'cross_below_sma': return prev_row[obv_col] > prev_row[obv_sma_col] and curr_row[obv_col] <= curr_row[obv_sma_col]
        elif indicator == 'OTT':
            p = params; ott_col = f"OTT_{p['length']}_{float(p['percent'])}"
            if pd.isna(prev_row[ott_col]) or pd.isna(curr_row[ott_col]): return False
            if condition == 'price_cross_above_ott': return prev_row['close'] < prev_row[ott_col] and curr_row['close'] >= curr_row[ott_col]
            if condition == 'price_cross_below_ott': return prev_row['close'] > prev_row[ott_col] and curr_row['close'] <= curr_row[ott_col]
    except (KeyError, TypeError): return False
    return False

def run_backtest(df: pd.DataFrame, strategy_info: dict, initial_cash: float = 10000.0):
    params = strategy_info.get('params', {})
    df_with_indicators = calculate_strategy_indicators(df, strategy_info)
    atr_col_name = f"ATRr_{ATR_PERIOD}"
    buy_rules, sell_rules = params.get('buy_rules', []), params.get('sell_rules', [])
    buy_quorum, sell_quorum = params.get('buy_quorum', len(buy_rules)), params.get('sell_quorum', len(sell_rules))
    cash, pos_size, buy_price = initial_cash, 0.0, 0.0; in_pos = False
    stop_loss_price = 0.0
    trades, win, closed = [], 0, 0; peak, max_dd = initial_cash, 0.0
    for i in range(1, len(df_with_indicators)):
        prev_row, curr_row = df_with_indicators.iloc[i-1], df_with_indicators.iloc[i]; price = curr_row['close']
        port_val = cash + (pos_size * price); peak = max(peak, port_val)
        dd = (peak - port_val) / peak if peak > 0 else 0; max_dd = max(max_dd, dd)
        if in_pos and USE_STOP_LOSS and stop_loss_price > 0 and curr_row['low'] <= stop_loss_price:
            sell_price = stop_loss_price
            cash, in_pos = pos_size * sell_price, False; closed += 1
            if sell_price > buy_price: win += 1
            trades.append({'type': 'STOP_LOSS', 'price': sell_price, 'date': curr_row['timestamp']})
            pos_size, buy_price, stop_loss_price = 0.0, 0.0, 0.0
            continue
        buy_votes = sum(_evaluate_rule(rule, prev_row, curr_row) for rule in buy_rules)
        if not in_pos and buy_votes >= buy_quorum:
            pos_size, buy_price, cash, in_pos = cash / price, price, 0.0, True
            trades.append({'type': 'BUY', 'price': price, 'date': curr_row['timestamp']})
            if USE_STOP_LOSS and atr_col_name in curr_row and pd.notna(curr_row[atr_col_name]):
                stop_loss_price = price - (curr_row[atr_col_name] * ATR_MULTIPLIER)
        sell_votes = sum(_evaluate_rule(rule, prev_row, curr_row) for rule in sell_rules)
        if in_pos and sell_votes >= sell_quorum:
            cash, in_pos = pos_size * price, False; closed += 1
            if price > buy_price: win += 1
            trades.append({'type': 'SELL', 'price': price, 'date': curr_row['timestamp']})
            pos_size, buy_price, stop_loss_price = 0.0, 0.0, 0.0
    final_val = cash if not in_pos else pos_size * df_with_indicators.iloc[-1]['close']
    net_profit = ((final_val - initial_cash) / initial_cash) * 100
    win_rate = (win / closed) * 100 if closed > 0 else 0.0
    print(f"--- BACKTEST SONUCU ---\nBitiş Bakiyesi: ${final_val:,.2f} | Net Kâr/Zarar: {net_profit:.2f}%\nKazanma Oranı: {win_rate:.2f}% | Maks. Düşüş: {max_dd*100:.2f}% | Toplam İşlem: {closed}")
    return {'net_profit_percent': round(net_profit, 2), 'total_trades': closed, 'win_rate': round(win_rate, 2), 'max_drawdown': round(max_dd * 100, 2)}

def generate_equity_curve(df: pd.DataFrame, strategy_info: dict, initial_cash: float):
    params = strategy_info.get('params', {})
    df_with_indicators = calculate_strategy_indicators(df, strategy_info)
    buy_rules, sell_rules = params.get('buy_rules', []), params.get('sell_rules', [])
    buy_quorum, sell_quorum = params.get('buy_quorum', len(buy_rules)), params.get('sell_quorum', len(sell_rules))
    cash, pos_size, buy_price = initial_cash, 0.0, 0.0; in_pos = False
    equity_over_time = []
    if not df_with_indicators.empty:
        equity_over_time.append({'timestamp': df_with_indicators.iloc[0]['timestamp'], 'portfolio_value': initial_cash})
    for i in range(1, len(df_with_indicators)):
        prev_row, curr_row = df_with_indicators.iloc[i-1], df_with_indicators.iloc[i]; price = curr_row['close']
        buy_votes = sum(_evaluate_rule(rule, prev_row, curr_row) for rule in buy_rules)
        if not in_pos and buy_votes >= buy_quorum:
            pos_size, buy_price, cash, in_pos = cash / price, price, 0.0, True
        sell_votes = sum(_evaluate_rule(rule, prev_row, curr_row) for rule in sell_rules)
        if in_pos and sell_votes >= sell_quorum:
            cash, in_pos = pos_size * price, False; pos_size, buy_price = 0.0, 0.0
        current_portfolio_value = cash + (pos_size * price)
        equity_over_time.append({'timestamp': curr_row['timestamp'], 'portfolio_value': current_portfolio_value})
    return pd.DataFrame(equity_over_time)

def check_live_signal(df: pd.DataFrame, strategy_info: dict) -> str:
    if len(df) < 100: return 'HOLD'
    df_with_indicators = calculate_strategy_indicators(df, strategy_info)
    params = strategy_info.get('params', {})
    if len(df_with_indicators) < 2: return 'HOLD'
    prev_row, curr_row = df_with_indicators.iloc[-2], df_with_indicators.iloc[-1]
    buy_rules, buy_quorum = params.get('buy_rules', []), params.get('buy_quorum', len(params.get('buy_rules', [])))
    if sum(_evaluate_rule(rule, prev_row, curr_row) for rule in buy_rules) >= buy_quorum: return 'BUY'
    sell_rules, sell_quorum = params.get('sell_rules', []), params.get('sell_quorum', len(params.get('sell_rules', [])))
    if sum(_evaluate_rule(rule, prev_row, curr_row) for rule in sell_rules) >= sell_quorum: return 'SELL'
    return 'HOLD'