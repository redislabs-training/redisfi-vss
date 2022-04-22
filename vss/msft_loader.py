from re import search
from datetime import timedelta
from time import perf_counter, sleep
from pickle import load
from random import triangular
from json import dumps, loads
from subprocess import Popen

import requests
from numpy import datetime64
from pandas import read_parquet, DatetimeIndex, DataFrame
from redis import Redis

import prefect
from prefect import task

from vss.db import set_filing_obj, set_embedding_on_filing_obj, semaphore, set_html_for_url, get_html_for_url

# "FT.CREATE" "filing:idx" "SCHEMA" "para_tag" "TEXT" "para_contents" "TEXT" "line_word_count" "TEXT" "COMPANY_NAME" "TEXT" "FILING_TYPE" "TEXT" "SIC_INDUSTRY" "TEXT" "DOC_COUNT" "NUMERIC" "CIK_METADATA" "NUMERIC" "all_capital" "NUMERIC" "FILED_DATE_YEAR" "NUMERIC" "FILED_DATE_MONTH" "NUMERIC" "FILED_DATE_DAY" "NUMERIC" "embedding" "VECTOR" "HNSW" "12" "TYPE" "FLOAT32" "DIM" "768" "DISTANCE_METRIC" "COSINE" "INITIAL_CAP" "150000" "M" "60" "EF_CONSTRUCTION" "500"
VECTOR_DIMENSIONS = 768
METADATA_NA_COLUMNS=['para_tag','COMPANY_NAME','SIC_INDUSTRY','SIC','FILING_TYPE']
METADATA_INDEX_COLUMNS=['para_tag','para_contents','line_word_count','COMPANY_NAME','FILING_TYPE','SIC_INDUSTRY','DOC_COUNT','CIK_METADATA','all_capital','FILED_DATE_YEAR','FILED_DATE_MONTH','FILED_DATE_DAY']
SEC_MAX_PER_SECOND = 5
SEC_URL_BASE = 'https://sec.gov/Archives/'
RATE_LIMIT_ATTEMPT_MAX = 120
MISSING_DOCS = ('edgar/data/1108524/0001108524-21-000014.txt', 'edgar/data/1108524/0001108524-20-000029.txt')

def _load_http_file_map():
    with open('data/filemap.json', 'r') as f:
        return loads(f.read())

HTTP_FILE_MAP = _load_http_file_map()

def download_data():
    
    with Popen(['wget', 'https://storage.googleapis.com/redisfi/data.tar', '-P', '/tmp']) as p:
        p.communicate()
        if p.returncode != 0:
            raise Exception('error downloading data')

    

                            ######################################
                            ## TASK I: Load metadata into Redis ##
                            ######################################   

@task(nout=2)
def load_metadata(metadata_file: str, redis_url: str, pipeline_interval: int) -> tuple:
    logger = prefect.context.get('logger')
    logger.info(f'getting data from {metadata_file}')
    metadata = _munge_metadata(read_parquet(metadata_file))
    
    data_map = {}
    data_map['offset'] = metadata.index.start
    data_map['records'] = metadata.to_dict('records')
    logger.info(f'file contained {len(data_map["records"])} records - transforming and loading into redis')
    _load_metadata_records(data_map, redis_url, pipeline_interval)
    
    file_key = metadata_file.split('_')[1].split('.')[0]

    return (file_key, data_map['offset'])

def _load_metadata_records(data_map: dict, redis_url: str, pipeline_interval: int):
        logger = prefect.context.get('logger')
        
        r = Redis.from_url(redis_url)
        start = perf_counter()  
        with r.pipeline(transaction=False) as pipe:
            offset = data_map['offset']
            counter = 0
            total_counter = 0
            batch_start = perf_counter()
            
            for _metadata in data_map['records']:    
                data = __build_object_from_row(_metadata)
                set_filing_obj(pipe, data, offset)
                
                if counter == pipeline_interval:
                    logger.debug('executing metadata batch')
                    pipe.execute()
                    batch_end = perf_counter()
                    logger.debug(f'metadata execute completed! {counter} records loaded to redis in {batch_end-batch_start:0.2f} seconds')
                    counter = 0
                    batch_start = perf_counter()

                offset += 1
                counter += 1
                total_counter += 1
            
            logger.debug('executing any remaining metadata records in pipeline')
            pipe.execute()
            batch_end = perf_counter()
            logger.debug(f'metadata execute completed! {counter} records loaded to redis in {batch_end-batch_start:0.2f} seconds')

        end = perf_counter()
        logger.info(f'work complete! {total_counter} records loaded to redis in {end-start:0.2f} seconds')

def _munge_metadata(metadata: DataFrame) -> DataFrame:
    metadata = __fill_nas(metadata)
    return metadata

def __build_object_from_row(row: dict) -> dict:

    obj = row

    obj['FILED_DATE'] = str(row['FILED_DATE'])
    obj['ACCEPTANCE_DATETIME'] = str(row['ACCEPTANCE_DATETIME'])
    obj['DATE_AS_OF_CHANGE'] = str(row['DATE_AS_OF_CHANGE'])
    obj['PERIOD'] = str(row['PERIOD'])
    obj['FISCAL_YEAR_END'] = str(row['FISCAL_YEAR_END'])
    obj['CIK'] = int(row['CIK'])
    obj['DOC_COUNT'] = int(row['DOC_COUNT'])
    obj['CIK_METADATA'] = int(row['CIK_METADATA'])
    obj['FILED_DATE_YEAR'] = int(row['FILED_DATE_YEAR'])
    obj['FILED_DATE_MONTH'] = int(row['FILED_DATE_MONTH'])
    obj['len_text'] = int(row['len_text'])
    obj['all_capital'] = int(row['all_capital'])
    obj['HTTP_FILE'] = HTTP_FILE_MAP[obj["FILE_NAME"]]

    return obj

def __fill_nas(metadata: DataFrame):
    #CLEAN UP NANs
    for c in METADATA_NA_COLUMNS:
        metadata[c] = metadata[c].fillna('N/A')

    #DT column clean up    
    metadata['FISCAL_YEAR_END'] = metadata['FISCAL_YEAR_END'].fillna(datetime64("1970-01-01 12:00:00"))
    metadata['PERIOD'] = metadata['PERIOD'].fillna(datetime64("1970-01-01 12:00:00"))
    metadata['FILED_DATE'] = metadata['FILED_DATE'].fillna(datetime64("1970-01-01 12:00:00"))
    metadata['ACCEPTANCE_DATETIME'] = metadata['ACCEPTANCE_DATETIME'].fillna(datetime64("1970-01-01 12:00:00"))
    metadata['DATE_AS_OF_CHANGE'] = metadata['DATE_AS_OF_CHANGE'].fillna(datetime64("1970-01-01 12:00:00"))

    #ADD FILED YEAR, MONTH & DAY
    dates = DatetimeIndex(metadata['FILED_DATE'])
    metadata['FILED_DATE_YEAR'] = dates.year
    metadata['FILED_DATE_MONTH'] = dates.month
    metadata['FILED_DATE_DAY'] = dates.day

    return metadata

                    ##############################################
                    ## TASK II: Enrich Redis Obj with Embedding ##
                    ##############################################

@task
def load_embeddings(args:tuple, redis_url: str, pipeline_interval: int):
    file_key, offset = args
    logger = prefect.context.get('logger')
    r = Redis.from_url(redis_url)
    start = perf_counter()
    filename = f'data/embeddings_{file_key}.pkl'
    logger.info(f'Opening embeddings file: {filename} | offset: {offset}')
    
    with open(filename, 'rb') as f:
        embeddings = load(f)

    logger.info(f'File contains {len(embeddings)} embeddings')

    with r.pipeline(transaction=False) as pipe:
        counter = 0
        total_counter = 0
        batch_start = perf_counter()
        for embedding in embeddings:
            set_embedding_on_filing_obj(pipe, offset, embedding)

            if counter == pipeline_interval:
                logger.debug('executing embedding batch')
                pipe.execute()
                batch_end = perf_counter()
                logger.debug(f'embedding execute completed! {counter} embeddings loaded to redis in {batch_end-batch_start:0.2f} seconds')
                counter = 0
                batch_start = perf_counter()

            counter += 1
            offset += 1
            total_counter += 1

        logger.debug('executing any remaining embeddings in pipeline')
        pipe.execute()
        batch_end = perf_counter()
        logger.debug(f'embedding execute completed! {counter} embeddings loaded to redis in {batch_end-batch_start:0.2f} seconds')
    
    
    end = perf_counter()
    logger.info(f'work complete! {total_counter} embeddings loaded to redis in {end-start:0.2f} seconds')

                                            ##############################
                                            ## CREATE FILENAME MAP FILE ##
                                            ##############################
@task 
def get_filenames_from_parquets(filename):
    df = read_parquet(filename)
    data = df.to_dict('records')

    filenames = set()
    for row in data:
        filenames.add(row['FILE_NAME'])

    return filenames

@task
def flatten_filename_sets(list_of_sets):
    grand_set = set()

    for _set in list_of_sets:
        grand_set = grand_set.union(_set)

    return list(grand_set)

@task(max_retries=5, retry_delay=timedelta(seconds=20), timeout=60)
def get_html_file_from_raw_file(raw_file_url: str, redis_url: str) -> tuple:
    logger = prefect.context.get('logger')
    
    if raw_file_url in MISSING_DOCS: # these raw files don't exist anymore
        return raw_file_url, '/'

    r = Redis.from_url(redis_url)
    html_url = get_html_for_url(r, raw_file_url)
    if html_url:
        return raw_file_url, html_url

    attempts = 0
    while True:
        if semaphore(r, SEC_MAX_PER_SECOND):
            logger.info(raw_file_url)
            resp = requests.get(SEC_URL_BASE + raw_file_url, 
                                headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'},
                                timeout=30, stream=True)

            if resp.status_code != 200:
                raise Exception(f'HTTP Call Failed: {resp}\n{resp.text}')

            filename = None
            for line in resp.iter_lines():
                match = search('<FILENAME>(.*)', line.decode('ascii'))
                if match:
                    filename = match.group(1)
                    break
            
            if not filename:
                raise Exception(f'filename not found in raw file: {raw_file_url}')

            url_parts, _file = raw_file_url.split('/')[0:-1], raw_file_url.split('/')[-1]
            url_parts.append(_file.split('.')[0].replace('-', ''))
            url_parts.append(filename)
            html_url = '/'.join(url_parts)
            set_html_for_url(r, raw_file_url, html_url)
            return raw_file_url, html_url
        else:
            if attempts < RATE_LIMIT_ATTEMPT_MAX:
                attempts += 1
                logger.info('waiting!')
                sleep(triangular(.25, 1.5))
            else:
                raise Exception('Max Attempts Reached')

@task
def write_filemap_file(file_map, file_location):
    file_map_dict = {}
    for raw, html in file_map:
        if type(raw) == bytes:
            raw = raw.decode('ascii')
        if type(html) == bytes:
            html = html.decode('ascii')

        file_map_dict[raw] = html

    with open(file_location, 'w') as f:
        f.write(dumps(file_map_dict))
