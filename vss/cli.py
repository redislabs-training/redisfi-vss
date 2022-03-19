from glob import glob
from time import perf_counter

from prefect import Flow, unmapped
from prefect.executors import DaskExecutor
from cleo import Application, Command

from vss.msft_loader import load_metadata, load_embeddings


class LoadCommand(Command):
    '''
    Load the VSS data into Redis

    load_msft
        {--redis-url=redis://localhost:6379 : Location of the Redis to Load to}
        {--pipeline-interval=50000 : Amount to break data load into for pipeline}
    '''
    def handle(self):

        metadata_files = glob('data/metadata*')
        pipeline_interval = int(self.option('pipeline-interval'))
        redis_url = self.option('redis-url')
        self.line(f'<info>Found</info> <comment>{len(metadata_files)}</comment> <info>metadata files</info>')
        start = perf_counter()
        

        with Flow('loader', executor=DaskExecutor(debug=True)) as flow:
            file_keys_and_offsets = load_metadata.map(*(metadata_files, unmapped(redis_url), unmapped(pipeline_interval)))
            load_embeddings.map(*(file_keys_and_offsets, unmapped(redis_url), unmapped(pipeline_interval)))

        self.info('<error>Handing off to Prefect/Dask</error>')
        flow.run()
        end = perf_counter()
        self.line(f'<info>Flow Completed! Total Execution Time:</info> <comment>{end-start:0.2f} seconds</comment>')


def run():
    app = Application(name='VSS')
    app.add(LoadCommand())
    app.run()