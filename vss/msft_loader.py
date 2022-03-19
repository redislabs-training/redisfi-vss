from time import perf_counter
from pickle import load
from numpy import datetime64, float32, ndarray
from pandas import read_parquet, DatetimeIndex, DataFrame
from redis import Redis

import prefect
from prefect import task

from vss.db import load_vss_obj, add_embedding_to_vss_obj

# "FT.CREATE" "vss:idx" "SCHEMA" "para_tag" "TEXT" "para_contents" "TEXT" "line_word_count" "TEXT" "COMPANY_NAME" "TEXT" "FILING_TYPE" "TEXT" "SIC_INDUSTRY" "TEXT" "DOC_COUNT" "NUMERIC" "CIK_METADATA" "NUMERIC" "all_capital" "NUMERIC" "FILED_DATE_YEAR" "NUMERIC" "FILED_DATE_MONTH" "NUMERIC" "FILED_DATE_DAY" "NUMERIC" "embedding" "VECTOR" "HNSW" "12" "TYPE" "FLOAT32" "DIM" "768" "DISTANCE_METRIC" "COSINE" "INITIAL_CAP" "150000" "M" "60" "EF_CONSTRUCTION" "500"
VECTOR_DIMENSIONS = 768
METADATA_NA_COLUMNS=['para_tag','COMPANY_NAME','SIC_INDUSTRY','SIC','FILING_TYPE']
METADATA_INDEX_COLUMNS=['para_tag','para_contents','line_word_count','COMPANY_NAME','FILING_TYPE','SIC_INDUSTRY','DOC_COUNT','CIK_METADATA','all_capital','FILED_DATE_YEAR','FILED_DATE_MONTH','FILED_DATE_DAY']


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

    _load_metadata_records(data_map, redis_url, pipeline_interval)
    
    file_key = metadata_file.split('_')[1].split('.')[0]

    return (file_key, data_map['offset'])
 
# @task
def _load_metadata_records(data_map: dict, redis_url: str, pipeline_interval: int):
        logger = prefect.context.get('logger')
        logger.info(f'{len(data_map["records"])} records')
        r = Redis.from_url(redis_url)
        start = perf_counter()
        with r.pipeline(transaction=False) as pipe:
            offset = data_map['offset']
            counter = 0
            batch_start = perf_counter()
            for _metadata in data_map['records']:
                
                data = __build_object_from_row(_metadata)
                load_vss_obj(pipe, data, offset)
                
                if counter == pipeline_interval:
                    logger.info('executing batch')
                    pipe.execute()
                    batch_end = perf_counter()
                    logger.info(f'execute completed! {counter} records loaded to redis in {batch_end-batch_start:0.2f} seconds')
                    counter = 0
                    batch_start = perf_counter()

                offset += 1
                counter += 1
            
            logger.info('executing any remaining records in pipeline')
            pipe.execute()
            batch_end = perf_counter()
            logger.info(f'execute completed! {counter} records loaded to redis in {batch_end-batch_start:0.2f} seconds')

        end = perf_counter()
        logger.info(f'work complete! {counter} records to redis loaded in {end-start:0.2f} seconds')

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
    # obj['embedding'] = _convert_embedding_to_bytes(embedding)

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
        batch_start = perf_counter()
        for embedding in embeddings:
            add_embedding_to_vss_obj(pipe, offset, _convert_embedding_to_bytes(embedding))

            if counter == pipeline_interval:
                logger.info('executing batch')
                pipe.execute()
                batch_end = perf_counter()
                logger.info(f'execute completed! {counter} embeddings loaded to redis in {batch_end-batch_start:0.2f} seconds')
                counter = 0
                batch_start = perf_counter()

            counter += 1
            offset += 1

        logger.info('executing any remaining embeddings in pipeline')
        pipe.execute()
        batch_end = perf_counter()
        logger.info(f'execute completed! {counter} embeddings loaded to redis in {batch_end-batch_start:0.2f} seconds')
    
    
    end = perf_counter()
    logger.info(f'work complete! {counter} embeddings loaded to redis in {end-start:0.2f} seconds')

    
    # return metadata_map

def _convert_embedding_to_bytes(embedding: ndarray):
    return embedding.astype(float32).tobytes()






# @task
# def create_index(redis_url: str, datatypes):
#     logger = prefect.context.get('logger')
#     logger.info(f'redis url: {redis_url}, datatypes: {datatypes}')
#     r = Redis.from_url(redis_url)
#     vss_index(r, METADATA_INDEX_COLUMNS, datatypes, VECTOR_DIMENSIONS)

# @task
# def batch_data(data_maps: dict, batch_size: int):
#     logger = prefect.context.get('logger')
#     logger.info(len(data_maps))
#     full = []
    
#     for data_map in data_maps:
#         records = data_map.pop('records')
#         embeddings = data_map.pop('embeddings')
#         batch = data_map.copy()

#         batch_records = []
#         counter = 0

#         for record, embedding in zip(records, embeddings):
#             batch_records.append(record)
#             if (counter + 1) % batch_size == 0:
#                 batch['records'] = batch_records
#                 batch['embeddings'] = embedding
#                 batch['offset'] += counter

#                 full.append(batch)
#                 batch = data_map.copy()
#                 batch_records = []

#             counter += 1
    
#     logger.info(f'created {len(full)} batches. {type(full[0])}')
#     return full
