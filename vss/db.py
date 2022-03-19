from redis import Redis
from redis.exceptions import ResponseError

_key_vss = lambda index: f'vss:{index}'

def vss_index(r: Redis, metadata_fields, datatypes, dimensions):
    idx =  r.ft(_key_vss('idx'))
    try:
       idx.info()
       return idx
    except ResponseError:
        pass

    create_command = ["FT.CREATE", _key_vss('idx'), "SCHEMA"]
    
    for field in metadata_fields:
        create_command.append(str(field))

        if str(datatypes[field]).lower().startswith('int'):
            create_command.append('NUMERIC')
        else:
            create_command.append('TEXT')
        
    create_command += ["embedding", "VECTOR", "HNSW", "12", "TYPE", "FLOAT32", "DIM", dimensions, "DISTANCE_METRIC", "COSINE",  "INITIAL_CAP", 150000, "M", 60, "EF_CONSTRUCTION", 500]
    r.execute_command(*create_command)

    return idx

def load_vss_obj(r: Redis, obj: dict, index: int):
    return r.hmset(_key_vss(index), obj)

def add_embedding_to_vss_obj(r: Redis, index: int, embedding: bytes):
    return r.hset(_key_vss(index), 'embedding', embedding)