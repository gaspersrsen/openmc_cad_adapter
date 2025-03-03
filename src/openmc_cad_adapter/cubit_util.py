_CUBIT_ID = 2


def reset_cubit_ids():
    global _CUBIT_ID
    _CUBIT_ID = 2


def lastid():
    global _CUBIT_ID
    id_out = _CUBIT_ID
    _CUBIT_ID += 1
    if _CUBIT_ID % int(1e4) == 0:
        print(f"Processed {_CUBIT_ID} actions")
    return id_out

def mul_id(n):
    global _CUBIT_ID
    id_out = _CUBIT_ID
    _CUBIT_ID += n
    if _CUBIT_ID % int(1e4) == 0:
        print(f"Processed {_CUBIT_ID} actions")
    return [id_out, _CUBIT_ID]


def new_variable():
    idn = lastid()
    return f"id{idn}"


def emit_get_last_id(type="body", cmds=None):
    idn = lastid()
    ids = f"id{idn}"
    if cmds is not None:
        cmds.append(f'#{{ {ids} = Id("{type}") }}')
    else:
        print('Warning: cmds is None')
    return ids

