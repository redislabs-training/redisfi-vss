from os import environ
from subprocess import Popen
from json import dumps
from flask import Flask, request
from redis import Redis, ResponseError
from sentence_transformers import SentenceTransformer

from vss import db as DB

MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
SEARCH_K = 1000
FACETS_K = 10000

app = Flask(__name__)
app.config['REDIS'] = Redis.from_url(environ.get('REDIS_URL', 'redis://localhost:6379'))
app.config['EXPORT_REDIS'] = Redis.from_url(environ.get('EXPORT_REDIS_URL', 'redis://localhost:6379'))

@app.route('/')
def search():
    _filter = request.args.get('filter')
    term = request.args.get('term')
    log_guid = request.args.get('log_guid')
    
    print(f'term: {term} | filter: {_filter}')

    if term is not None:
        term = get_embedding(term)
    try:
        results, total, duration = DB.query_filings(app.config['REDIS'], term, _filter, SEARCH_K, log_guid=log_guid, export_redis=app.config['EXPORT_REDIS'])
        ret = {'results':results, 'metrics':{'duration':duration, 'total':total}}
    except ResponseError:
        import traceback
        traceback.print_exc()
        ret = {'results':[] , 'metrics':{'duration':0, 'total':0}}

    return dumps(ret)

@app.route('/facets')
def facets():
    term = request.args.get('term')
    _filter = request.args.get('filter')
    _facets = DB.get_facets_for_term(app.config['REDIS'], term, _filter)
    if _facets is not None:
        return _facets
    try:
        vector = get_embedding(term)
        results, _, _ = DB.query_filings(app.config['REDIS'], vector=vector, _filter=_filter, k=FACETS_K)
    except ResponseError:
        results = []

    _facets = {}
    for result in results:
        _facets[result['COMPANY_NAME']] = _facets.get(result['COMPANY_NAME'], 0) + 1
    
    DB.set_facets_for_term(app.config['REDIS'], term, _filter, _facets)

    return _facets

@app.route('/healthcheck')
def healthcheck():
    return str(int(app.config['REDIS'].get('vss-loader') or b'0'))

def get_embedding(term: str):
    embedding = DB.get_embedding_for_term(app.config['REDIS'], term)
    if embedding is not None:
        return embedding
    
    embedding = MODEL.encode(term)

    DB.set_embedding_for_term(app.config['REDIS'], term, embedding)  

    return embedding
   
def run(debug=False, redis_url='redis://', export_redis_url='redis://'):
    # This is ultimately a hack around the way the flask debug server works
    # and a way of baking the overall gunicorn run command into the CLI.
    #
    # Take config from CLI (cleo) and plug it into env vars, so the flask
    # app itself is pulling config from env, but that's being driven from 
    # the CLI and handed off here

    env = environ.copy()
    env['REDIS_URL'] = redis_url
    env['EXPORT_REDIS_URL'] = export_redis_url

    if debug:
        with Popen(['poetry', 'run', 'python3', 'vss/wsapi.py'], env=env) as _app:
            _app.communicate()
    else:
        with Popen(['poetry', 'run', 'gunicorn', '-b', '0.0.0.0:7777', 'vss.wsapi:app'], env=env) as _app:
            _app.communicate()

if __name__ == '__main__':
    app.run(debug=True, port=7777)