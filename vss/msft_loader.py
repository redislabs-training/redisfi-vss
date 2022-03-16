from pickle import load
from numpy import datetime64, float32, ndarray
from pandas import read_parquet, DatetimeIndex, DataFrame


METADATA_NA_COLUMNS=['para_tag','COMPANY_NAME','SIC_INDUSTRY','SIC','FILING_TYPE']

def build_object_from_row(row: dict, embedding: bytes) -> dict:

    obj = row

    obj['FILED_DATE']=str(row['FILED_DATE'])
    obj['ACCEPTANCE_DATETIME']=str(row['ACCEPTANCE_DATETIME'])
    obj['DATE_AS_OF_CHANGE']=str(row['DATE_AS_OF_CHANGE'])
    obj['PERIOD']=str(row['PERIOD'])
    obj['FISCAL_YEAR_END']=str(row['FISCAL_YEAR_END'])
    obj['CIK']=int(row['CIK'])
    obj['DOC_COUNT']=int(row['DOC_COUNT'])
    obj['CIK_METADATA']=int(row['CIK_METADATA'])
    obj['FILED_DATE_YEAR']=int(row['FILED_DATE_YEAR'])
    obj['FILED_DATE_MONTH']=int(row['FILED_DATE_MONTH'])
    obj['len_text']=int(row['len_text'])
    obj['all_capital']=int(row['all_capital'])
    obj['embedding'] = _convert_embedding_to_bytes(embedding)

    return obj

def load_embeddings(embeddings_file: str):
    with open(embeddings_file, 'rb') as f:
        return load(f)

def load_metadata(metadata_file: str) -> DataFrame:
    return read_parquet(metadata_file)

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

