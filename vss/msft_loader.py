from time import perf_counter
from pickle import load
from numpy import datetime64, float32, ndarray
from pandas import read_parquet, DatetimeIndex, DataFrame
from redis import Redis

import prefect
from prefect import task

from vss.db import vss_index, load_vss_obj

# "FT.CREATE" "vss:idx" "SCHEMA" "para_tag" "TEXT" "para_contents" "TEXT" "line_word_count" "TEXT" "COMPANY_NAME" "TEXT" "FILING_TYPE" "TEXT" "SIC_INDUSTRY" "TEXT" "DOC_COUNT" "NUMERIC" "CIK_METADATA" "NUMERIC" "all_capital" "NUMERIC" "FILED_DATE_YEAR" "NUMERIC" "FILED_DATE_MONTH" "NUMERIC" "FILED_DATE_DAY" "NUMERIC" "embedding" "VECTOR" "HNSW" "12" "TYPE" "FLOAT32" "DIM" "768" "DISTANCE_METRIC" "COSINE" "INITIAL_CAP" "150000" "M" "60" "EF_CONSTRUCTION" "500"
VECTOR_DIMENSIONS = 768
METADATA_NA_COLUMNS=['para_tag','COMPANY_NAME','SIC_INDUSTRY','SIC','FILING_TYPE']
METADATA_INDEX_COLUMNS=['para_tag','para_contents','line_word_count','COMPANY_NAME','FILING_TYPE','SIC_INDUSTRY','DOC_COUNT','CIK_METADATA','all_capital','FILED_DATE_YEAR','FILED_DATE_MONTH','FILED_DATE_DAY']

@task
def create_index(redis_url: str, datatypes):
    logger = prefect.context.get('logger')
    logger.info(f'redis url: {redis_url}, datatypes: {datatypes}')
    r = Redis.from_url(redis_url)
    vss_index(r, METADATA_INDEX_COLUMNS, datatypes, VECTOR_DIMENSIONS)

@task
def batch_data(data_maps: dict, batch_size: int):
    logger = prefect.context.get('logger')
    logger.info(len(data_maps))
    full = []
    
    for data_map in data_maps:
        records = data_map.pop('records')
        embeddings = data_map.pop('embeddings')
        batch = data_map.copy()

        batch_records = []
        counter = 0

        for record, embedding in zip(records, embeddings):
            batch_records.append(record)
            if (counter + 1) % batch_size == 0:
                batch['records'] = batch_records
                batch['embeddings'] = embedding
                batch['offset'] += counter

                full.append(batch)
                batch = data_map.copy()
                batch_records = []

            counter += 1
    
    logger.info(f'created {len(full)} batches. {type(full[0])}')
    return full

@task
def load_data(data_map: dict, redis_url: str, pipeline_interval: int):
        logger = prefect.context.get('logger')
        logger.info(f'{len(data_map["records"])} records')
        r = Redis.from_url(redis_url)
        start = perf_counter()
        with r.pipeline(transaction=False) as pipe:
            offset = data_map['offset']
            counter = 1
            for _metadata, embedding in zip(data_map['records'], data_map['embeddings']):
                batch_start = perf_counter()
                data = build_object_from_row(_metadata, embedding)
                load_vss_obj(pipe, data, offset)
                
                if counter % pipeline_interval == 0:
                    logger.info('executing batch')
                    pipe.execute()
                    batch_end = perf_counter()
                    logger.info(f'{counter} loaded in {batch_end-batch_start:0.2f} seconds')

                offset += 1
                counter += 1

            pipe.execute()

        end = perf_counter()
        logger.info(f'{counter} records loaded in {end-start:0.2f} seconds')


def build_object_from_row(row: dict, embedding: bytes) -> dict:

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
    obj['embedding'] = _convert_embedding_to_bytes(embedding)

    return obj

@task
def get_embeddings(metadata_map: dict):
    logger = prefect.context.get('logger')
    logger.info(metadata_map.keys())
    filename = f'data/embeddings_{metadata_map["file_key"]}.pkl'
    
    with open(filename, 'rb') as f:
        metadata_map['embeddings'] = load(f)
    
    return metadata_map

@task
def get_metadata(metadata_file: str) -> dict:
    logger = prefect.context.get('logger')
    logger.info(f'getting data from {metadata_file}')
    metadata = munge_metadata(read_parquet(metadata_file))
    
    ret = {}
    ret['offset'] = metadata.index.start
    ret['records'] = metadata.to_dict('records')
    ret['file_key'] = metadata_file.split('_')[1].split('.')[0]
    ret['datatypes'] = metadata.dtypes


    return ret


def munge_metadata(metadata: DataFrame) -> DataFrame:
    metadata = _fill_nas(metadata)
    return metadata

def _convert_embedding_to_bytes(embedding: ndarray):
    return embedding.astype(float32).tobytes()

def _fill_nas(metadata: DataFrame):
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

