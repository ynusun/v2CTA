def get_rules():
    from config import PARAMETER_SPACES
    rules = {'buy': [], 'sell': []}
    ott_params = PARAMETER_SPACES.get('OTT', {})
    for l in ott_params.get('length', [2]):
        for p in ott_params.get('percent', [1.4]):
            params = {'length': l, 'percent': float(p), 'mav': 'VAR'}
            rule_name_base = f"OTT_{l}_{float(p)}"
            rules['buy'].append({'name': f"{rule_name_base}_Buy", 'rule': {'indicator': 'OTT', 'condition': 'price_cross_above_ott', 'params': params, 'regimes': ['Bull', 'Bear']}})
            rules['sell'].append({'name': f"{rule_name_base}_Sell", 'rule': {'indicator': 'OTT', 'condition': 'price_cross_below_ott', 'params': params, 'regimes': ['Bull', 'Bear']}})
    return rules