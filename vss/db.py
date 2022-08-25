from time import time
from json import dumps

from numpy import ndarray, float32

from redis import Redis
from redis.commands.search.query import Query
from redis.commands.json.path import Path
from redis.commands.search.commands import SEARCH_CMD, SearchCommands

RETURN_FIELDS = ('COMPANY_NAME','para_contents','FILED_DATE', "FILE_NAME", "HTTP_FILE", "FILING_TYPE")

_key_commands    = lambda guid: f'commands:{guid}'
_key_filing = lambda index: f'filing:{index}'
_key_term_facets = lambda term, _filter: f'term:{term}:{_filter if _filter else ""}:facets'
_key_term_vector = lambda term: f'term:{term}:vector'
_key_semaphore = lambda: f'semaphore:{int(time())}'
_key_url = lambda url: f'url:{url}'

def _convert_embedding_to_bytes(embedding: ndarray):
    if type(embedding) == bytes:
        return embedding
    else:
        return embedding.astype(float32).tobytes()

def _build_search_query(index: SearchCommands, query: Query, args=None):
    return ' '.join([SEARCH_CMD] + list(map(str, index._mk_query_args(query, args)[0])))

def semaphore(r: Redis, max: int):
    key = _key_semaphore()
    if r.incr(key) > max:
        return False
    
    if r.ttl(key) == -1:
        r.expire(key, 1)

    return True

def get_html_for_url(r: Redis, url: str):
    return r.get(_key_url(url))

def get_facets_for_term(r: Redis, term: str, _filter: str):
    return r.json().get(_key_term_facets(term, _filter))

def set_facets_for_term(r: Redis, term: str, _filter: str, obj: dict):
    return r.json().set(_key_term_facets(term, _filter), Path.root_path(), obj)

def get_embedding_for_term(r: Redis, term: str):
    return r.get(_key_term_vector(term))

def get_html_for_url(r: Redis, url: str):
    return r.get(_key_url(url))

def set_embedding_for_term(r: Redis, term: str, embedding: ndarray):
    return r.set(_key_term_vector(term), _convert_embedding_to_bytes(embedding))

def set_filing_obj(r: Redis, obj: dict, index: int):
    return r.hmset(_key_filing(index), obj)

def set_embedding_on_filing_obj(r: Redis, index: int, embedding: ndarray):
    return r.hset(_key_filing(index), 'embedding', _convert_embedding_to_bytes(embedding))

def set_html_for_url(r: Redis, raw_url: str, html_url: str):
    return r.set(_key_url(raw_url), html_url)

def query_filings(r: Redis, vector=None, _filter=None, k=10, log_guid=None, export_redis=None):
    if _filter is None and vector is not None:
        # only a vector to search for
        query_str = f'*=>[KNN $K @embedding $VECTOR]'
        vector_bytes = _convert_embedding_to_bytes(vector)
        params = {'K':k, 'VECTOR':vector_bytes}
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
        query_str = f'({_filter})=>[KNN $K @embedding $VECTOR]'
        vector_bytes = _convert_embedding_to_bytes(vector)
        params = {'K':k, 'VECTOR':vector_bytes}
        sort_by = '__embedding_score'
        asc = True
    
    idx = r.ft(_key_filing('idx'))

    q = Query(query_str).paging(0, k).sort_by(sort_by, asc=asc).return_fields(*RETURN_FIELDS).dialect(2)
    results = idx.search(q, params)
    if export_redis is None:
        export_redis = r

    query = _build_search_query(idx, q, params)
    if vector is not None:
        query = query.replace(str(vector_bytes), '&lt;vector_bytes&gt;')

    set_or_print_commands(export_redis, log_guid, query, results.duration)

    return [result.__dict__ for result in results.docs], len(results.docs), results.duration

def set_or_print_commands(redis: Redis, guid: str, command: str, time=0):
    print(command)
    if guid is not None:
        if type(command) is not str:
            command = dumps(command)
        redis.xadd(_key_commands(guid), {'command':command, 'time':time})
    else:
        print(command)