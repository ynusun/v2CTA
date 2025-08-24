def get_rules():
    from config import PARAMETER_SPACES
    rules = {'buy': [], 'sell': []}
    for period in PARAMETER_SPACES['RSI']['rsi_period']:
        params = {'rsi_period': period}
        rules['buy'].append({'name': f'RSI_{period}_Buy', 'rule': {'indicator': 'RSI', 'condition': 'cross_above', 'value': 30, 'params': params, 'regimes': ['Sideways', 'Bull']}})
        rules['sell'].append({'name': f'RSI_{period}_Sell', 'rule': {'indicator': 'RSI', 'condition': 'cross_below', 'value': 70, 'params': params, 'regimes': ['Sideways', 'Bear']}})
    return rules