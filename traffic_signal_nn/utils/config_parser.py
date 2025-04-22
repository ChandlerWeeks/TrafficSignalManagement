import configparser


def _auto(val: str):
    for caster in (int, float):
        try:
            return caster(val)
        except ValueError:
            pass
    if val.lower() in ('true', 'false'):
        return val.lower() == 'true'
    return val


def load_config(path: str) -> dict:
    """
    INI → {SECTION: {KEY: value}}
    """
    p = configparser.ConfigParser()
    p.read(path, encoding='utf‑8')
    out = {}
    for sec in p.sections():
        out[sec.upper()] = {k.upper(): _auto(v) for k, v in p[sec].items()}
    return out
