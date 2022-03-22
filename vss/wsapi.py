from os import environ
from subprocess import Popen
from json import dumps, load, loads
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from flask import Flask, request
from redis import Redis

from vss import db as DB

SCORE_URL = 'https://mpnetemb.eastus2.inference.ml.azure.com/score'
SCORE_API_KEY = 'Qimsm7lN1kTWdS7heH2MbT9HPuo7vyOv'
K = 5
LIMIT_DEFAULT = 1000
FACET_LIMIT = 1000000

app = Flask(__name__)
app.config['REDIS'] = Redis.from_url(environ.get('REDIS_URL', 'redis://localhost:6379'))

@app.route('/')
def search():
    _filter = request.args.get('filter')
    limit = request.args.get('limit', LIMIT_DEFAULT)
    term = request.args.get('term')
    
    print(f'term: {term} | filter: {_filter}')

    if term is not None:
        term = get_embedding(term)

    results, total, duration = DB.query_filings(app.config['REDIS'], term, _filter, K, limit)
    ret = {'results':results, 'metrics':{'duration':duration, 'total':total}}
    return dumps(ret)

@app.route('/facets')
def facets():
    term = request.args.get('term')
    _facets = DB.get_facets_for_term(app.config['REDIS'], term)
    if _facets is not None:
        return _facets

    vector = get_embedding(term)
    results, _, _ = DB.query_filings(app.config['REDIS'], vector=vector, k=K, limit=FACET_LIMIT)

    _facets = {}
    for result in results:
        _facets[result['COMPANY_NAME']] = _facets.get(result['COMPANY_NAME'], 0) + 1
    
    DB.set_facets_for_term(app.config['REDIS'], term, _facets)

    return _facets

@app.route('/healthcheck')
def healthcheck():
    return str(app.config['REDIS'].ping())

def get_embedding(term: str):
    embedding = DB.get_embedding_for_term(app.config['REDIS'], term)
    if embedding is not None:
        return embedding

    ## This doesn't work with requests and I couldn't figure out why
    headers = {'Content-Type':'application/json', 'Authorization':f'Bearer {SCORE_API_KEY}'}
    data = {'data':term}
    body = str.encode(dumps(data))
    req = Request(SCORE_URL, body, headers)

    try:
        response = urlopen(req)
        embedding = np.fromstring(load(response)[1:-1], dtype=np.float32, sep=',')
        
        DB.set_embedding_for_term(app.config['REDIS'], term, embedding)
        return embedding
    except HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(loads(error.read().decode("utf8", 'ignore')))

def run(debug=False, redis_url='redis://'):
    # This is ultimately a hack around the way the flask debug server works
    # and a way of baking the overall gunicorn run command into the CLI.
    #
    # Take config from CLI (cleo) and plug it into env vars, so the flask
    # app itself is pulling config from env, but that's being driven from 
    # the CLI and handed off here

    env = environ.copy()
    env['REDIS_URL'] = redis_url

    if debug:
        with Popen(['poetry', 'run', 'python3', 'vss/wsapi.py'], env=env) as _app:
            _app.communicate()
    else:
        with Popen(['poetry', 'run', 'gunicorn', '-b', '0.0.0.0:7777', 'vss.wsapi:app'], env=env) as _app:
            _app.communicate()

if __name__ == '__main__':
    app.run(debug=True, port=7777)