from os import environ
from glob import glob
from time import perf_counter

from prefect import Flow, unmapped
from prefect.executors import DaskExecutor
from cleo import Application, Command

from vss.msft_loader import load_metadata, load_embeddings
from vss.wsapi import run as run_wsapi


class LoadCommand(Command):
    '''
    Load the VSS data into Redis

    load
        {--r|redis-url=redis://localhost:6379 : Location of the Redis to Load to}
        {--pipeline-interval=50000 : Amount to break data load into for pipeline}
        {--embedding-pipeline-reduction-denominator=3 : Amount to divide pipeline by for embedding load}
    '''
    def handle(self):

        metadata_files = glob('data/metadata*')
        pipeline_interval = int(self.option('pipeline-interval'))
        reduction_factor = int(self.option('embedding-pipeline-reduction-denominator'))

        redis_url = environ.get('REDIS_URL', self.option('redis-url'))

        self.line(f'<info>Found</info> <comment>{len(metadata_files)}</comment> <info>metadata files</info>')
        start = perf_counter()
    
        with Flow('loader', executor=DaskExecutor()) as flow:
            file_keys_and_offsets = load_metadata.map(*(metadata_files, unmapped(redis_url), unmapped(pipeline_interval)))
            load_embeddings.map(*(file_keys_and_offsets, unmapped(redis_url), unmapped(pipeline_interval/reduction_factor)))

        self.line('<error>Handing off to Prefect/Dask</error>')
        flow.run()
        end = perf_counter()
        self.line(f'<info>Flow Completed! Total Execution Time:</info> <comment>{end-start:0.2f} seconds</comment>')

class RunCommand(Command):
    '''
    Run the VSS microservice.

    run
        {--debug : Runs the Debug Server}
        {--H|redis-url=redis://localhost:6379 : Redis URL - can also set with REDIS_URL env var}
    '''
    def handle(self):
        debug = self.option('debug')
        redis_url = environ.get('REDIS_URL', self.option('redis-url'))
        self.line(f'<info>Redis URL:</info> <comment>{redis_url}</comment>')
        
        run_wsapi(debug=debug, redis_url=redis_url)


def run():
    app = Application(name='VSS')
    app.add(LoadCommand())
    app.add(RunCommand())
    app.run()