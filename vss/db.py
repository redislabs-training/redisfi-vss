from redis import Redis
from redis.exceptions import ResponseError

_key_vss = lambda index: f'vss:{index}'

def create_index(r: Redis, metadata_fields):
    idx =  r.ft(_key_vss('idx'))
    try:
       idx.info()
    except ResponseError:
        pass

def load_vss_obj(r: Redis, obj: dict, index: int):
    return r.hmset(_key_vss(index), obj)