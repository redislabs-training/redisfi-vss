from numpy import ndarray, float32
from redis import Redis
from redis.commands.search.query import Query

_key_vss = lambda index: f'vss:{index}'

def load_vss_obj(r: Redis, obj: dict, index: int):
    return r.hmset(_key_vss(index), obj)

def add_embedding_to_vss_obj(r: Redis, index: int, embedding: ndarray):
    return r.hset(_key_vss(index), 'embedding', _convert_embedding_to_bytes(embedding))

def query(r: Redis, vector: ndarray=None, _filter=None, k=10):
    if _filter is None and vector is not None:
        # only a vector to search for
        vector_bytes = _convert_embedding_to_bytes(vector)
        params = {'K':k, 'VECTOR':vector_bytes}
        query_str = f'*=>[TOP_K $K @embedding $VECTOR]'
        sort_by = '__embedding_score'
        asc = True

    elif vector is None:
        # no vector to search for 
        query_str = _filter
        params = None
        sort_by = 'FILED_DATE_YEAR'
        asc = False
    else:
        # search for both
        query_str = f'({_filter})=>[TOP_K $K @embedding $VECTOR]'
        vector_bytes = _convert_embedding_to_bytes(vector)
        params = {'K':k, 'VECTOR':vector_bytes}
        sort_by = '__embedding_score'
        asc = True

    q = Query(query_str).paging(0, k).sort_by(sort_by, asc=asc).return_fields('COMPANY_NAME','para_contents','FILED_DATE', "FILE_NAME")
    results = r.ft(_key_vss('idx')).search(q, params)
    return [result.__dict__ for result in results.docs], results.total, results.duration

def _convert_embedding_to_bytes(embedding: ndarray):
    return embedding.astype(float32).tobytes()

# def vss_index(r: Redis, metadata_fields, datatypes, dimensions):
#     idx =  r.ft(_key_vss('idx'))
#     try:
#        idx.info()
#        return idx
#     except ResponseError:
#         pass

#     # create_command = ["FT.CREATE", _key_vss('idx'), "SCHEMA"]
    
#     # for field in metadata_fields:
#     #     create_command.append(str(field))

#     #     if str(datatypes[field]).lower().startswith('int'):
#     #         create_command.append('NUMERIC')
#     #     else:
#     #         create_command.append('TEXT')
        
#     # create_command += ["embedding", "VECTOR", "HNSW", "12", "TYPE", "FLOAT32", "DIM", dimensions, "DISTANCE_METRIC", "COSINE",  "INITIAL_CAP", 150000, "M", 60, "EF_CONSTRUCTION", 500]
#     # r.execute_command(*create_command)

#     return idx