def feature(data):
    data['is_precip'] = [1 if x>0 else 0 for x in data['precipcover']]
    return data
