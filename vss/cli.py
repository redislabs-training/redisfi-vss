from os import environ
from glob import glob
from time import perf_counter, sleep

from prefect import Flow, unmapped
from prefect.executors import DaskExecutor
from cleo import Application, Command
from clikit.api.io.flags import DEBUG
from redis.exceptions import ConnectionError

from vss.msft_loader import (load_metadata,
                             load_embeddings, 
                             get_filenames_from_parquets, 
                             flatten_filename_sets, 
                             get_html_file_from_raw_file,
                             write_filemap_file,
                             download_data,
                             create_index, 
                             mark_loader_started,
                             mark_loader_completed,
                             mark_loader_failed)

from vss.wsapi import run as run_wsapi

class CreateHTMLFileMap(Command):
    '''
    Go get the HTML file names for the reports from the raw text reports 

    create_filemap
        {--r|redis-url=redis://localhost:6379 : Location of the Redis to Load to - can also set with VSS_REDIS_URL env var}
        {--o|output-location=data/filemap.json : Location to store the output file}
    '''
    def handle(self):
        output_location = self.option('output-location')
        map_file = glob(output_location)
        if map_file and not self.confirm(f'Existing Map File Found at {output_location} - Recreate?', False):
            self.info('Ok, thanks. Have a good day!')
            return
    
        metadata_files = glob('data/metadata*')
        self.line(f'<info>Found</info> <comment>{len(metadata_files)}</comment> <info>metadata files</info>')
        redis_url = self.option('redis-url')

        with Flow('filemap', executor=DaskExecutor()) as flow:
            filenames_batched = get_filenames_from_parquets.map(metadata_files)
            filenames_flattened = flatten_filename_sets(filenames_batched)
            file_map = get_html_file_from_raw_file.map(filenames_flattened, unmapped(redis_url))
            write_filemap_file(file_map, output_location)

        flow.run()

        #get_html_file_from_raw_file.run('edgar/data/882184/0000882184-19-000022.txt', redis_url)

class LoadCommand(Command):
    '''
    Load the VSS data into Redis

    load
        {--r|redis-url=redis://localhost:6379 : Location of the Redis to Load to - can also set with VSS_REDIS_URL env var}
        {--pipeline-interval=50000 : Amount to break data load into for pipeline}
        {--reduction-factor=3 : Amount to divide pipeline by for embedding load}
        {--retry-count=20 : Number of times to retry redis for index creation}
    '''
    def handle(self):

        metadata_files = glob('data/metadata*')
    
        if not metadata_files:
            if self.confirm('data files missing. Download? (7.3GB)', True):
                download_data()
                metadata_files = glob('data/metadata*')
            else:
                self.line('Ok, Have a Good Day!')
                return 1
                
        pipeline_interval = int(self.option('pipeline-interval'))
        reduction_factor = int(self.option('reduction-factor'))

        redis_url = environ.get('VSS_REDIS_URL', self.option('redis-url'))
        retry_count = int(self.option('retry-count'))
        with self.spin(f'<info>Connecting to Redis @ <comment>{redis_url}</info>', f'<info>Connected to Redis @ <comment>{redis_url}</info>'):
            success = False
            tries = 0
            while not success:
                try:
                    create_index(redis_url)
                    success = True
                except ConnectionError as e:
                    self.line(f'<error>Error creating index: {e}</error>', verbosity=DEBUG)
                    tries += 1
                    self.line(f'<error>Waiting 10 seconds and trying again - {tries} of {retry_count}</error>', verbosity=DEBUG)
                    
                    if tries >= retry_count:
                        self.line(f'\n<error>Failed to connect to Redis after {retry_count} tries</error>')
                        raise
                    
                    sleep(10)
                    

        
        self.info('Index Created!')
        self.line(f'<info>Found</info> <comment>{len(metadata_files)}</comment> <info>metadata files</info>')
        mark_loader_started(redis_url)
        with Flow('loader', executor=DaskExecutor()) as flow:
            file_keys_and_offsets = load_metadata.map(*(metadata_files, unmapped(redis_url), unmapped(pipeline_interval)))
            load_embeddings.map(*(file_keys_and_offsets, unmapped(redis_url), unmapped(pipeline_interval/reduction_factor)))

        self.line('<error>Handing off to Prefect/Dask</error>')
        start = perf_counter()
        result = flow.run()
        end = perf_counter()
        
        if result.is_successful():
            mark_loader_completed(redis_url)
        else:
            mark_loader_failed(redis_url)

        self.line(f'<info>Flow Completed! Total Execution Time:</info> <comment>{end-start:0.2f} seconds</comment>')

class RunCommand(Command):
    '''
    Run the VSS microservice.

    run
        {--debug : Runs the Debug Server}
        {--redis-url=redis://localhost:6379 : Redis URL - can also set with VSS_REDIS_URL env var}
        {--command-export-redis-url=redis://localhost:6379 : Redis URL for Command exports - can also set with REDIS_URL env var}
    '''
    def handle(self):
        debug = self.option('debug')
        redis_url = environ.get('VSS_REDIS_URL', self.option('redis-url'))
        export_redis_url = environ.get('REDIS_URL', self.option('command-export-redis-url'))
        self.line(f'<info>Redis URL:</info> <comment>{redis_url}</comment>')
        self.line(f'<info>Export Redis URL:</info> <comment>{export_redis_url}</comment>')
        
        run_wsapi(debug=debug, redis_url=redis_url, export_redis_url=export_redis_url)


def run():
    app = Application(name='VSS')
    app.add(LoadCommand())
    app.add(RunCommand())
    app.add(CreateHTMLFileMap())
    app.run()