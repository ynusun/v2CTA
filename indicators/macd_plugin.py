def get_rules():
    from config import PARAMETER_SPACES
    rules = {'buy': [], 'sell': []}
    for p_set in PARAMETER_SPACES['MACD']:
        name_suffix = f"_{p_set['fast_period']}_{p_set['slow_period']}_{p_set['signal_period']}"
        rules['buy'].append({'name': f'MACD{name_suffix}_Buy', 'rule': {'indicator': 'MACD', 'condition': 'cross_above_signal', 'params': p_set, 'regimes': ['Bull', 'Bear']}})
        rules['sell'].append({'name': f'MACD{name_suffix}_Sell', 'rule': {'indicator': 'MACD', 'condition': 'cross_below_signal', 'params': p_set, 'regimes': ['Bull', 'Bear']}})
    return rules