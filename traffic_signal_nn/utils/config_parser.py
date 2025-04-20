# traffic_signal_nn/utils/config_parser.py

import configparser

def load_config(path):
    """
    Read INI file and return a dict-of-dicts: {SECTION: {KEY: value, …}, …}
    Values are auto-cast to int/float/bool when possible.
    """
    parser = configparser.ConfigParser()
    parser.read(path)
    cfg = {}
    for section in parser.sections():
        d = {}
        for key, val in parser[section].items():
            d[key.upper()] = _auto_cast(val)
        cfg[section.upper()] = d
    return cfg

def _auto_cast(val):
    for caster in (int, float):
        try:
            return caster(val)
        except ValueError:
            pass
    low = val.lower()
    if low in ("true","false"):
        return low == "true"
    return val
