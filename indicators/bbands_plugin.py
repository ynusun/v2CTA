def get_rules():
    from config import PARAMETER_SPACES
    rules = {'buy': [], 'sell': []}
    for std in PARAMETER_SPACES['BBANDS']['std']:
        params = {'length': 20, 'std': std}
        rules['buy'].append({'name': f'BBands_{std}_Buy', 'rule': {'indicator': 'BBANDS', 'condition': 'price_cross_above_lower', 'params': params, 'regimes': ['Sideways']}})
        rules['sell'].append({'name': f'BBands_{std}_Sell', 'rule': {'indicator': 'BBANDS', 'condition': 'price_cross_below_upper', 'params': params, 'regimes': ['Sideways']}})
    return rules