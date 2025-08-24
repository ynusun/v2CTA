def get_rules():
    from config import PARAMETER_SPACES
    rules = {'buy': [], 'sell': []}
    for fast in PARAMETER_SPACES['MA_CROSS']['fast_period']:
        for slow in PARAMETER_SPACES['MA_CROSS']['slow_period']:
            if fast >= slow: continue
            params = {'fast_period': fast, 'slow_period': slow}
            rules['buy'].append({'name': f'MA_{fast}_{slow}_Buy', 'rule': {'indicator': 'MA_CROSS', 'condition': 'cross_above', 'params': params, 'regimes': ['Bull', 'Bear']}})
            rules['sell'].append({'name': f'MA_{fast}_{slow}_Sell', 'rule': {'indicator': 'MA_CROSS', 'condition': 'cross_below', 'params': params, 'regimes': ['Bull', 'Bear']}})
    return rules