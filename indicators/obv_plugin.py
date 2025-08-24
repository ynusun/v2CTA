def get_rules():
    from config import PARAMETER_SPACES
    rules = {'buy': [], 'sell': []}
    for p_set in PARAMETER_SPACES['OBV']:
        period = p_set['obv_sma_period']
        rules['buy'].append({'name': f'OBV_{period}_Buy', 'rule': {'indicator': 'OBV', 'condition': 'cross_above_sma', 'params': p_set, 'regimes': ['Bull', 'Bear', 'Sideways']}})
        rules['sell'].append({'name': f'OBV_{period}_Sell', 'rule': {'indicator': 'OBV', 'condition': 'cross_below_sma', 'params': p_set, 'regimes': ['Bull', 'Bear', 'Sideways']}})
    return rules