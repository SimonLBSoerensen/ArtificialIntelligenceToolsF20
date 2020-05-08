def save_to_pickle(filename, var):
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        var = pickle.load(handle)
    return var


def save_json(filename, data):
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)