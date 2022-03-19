from glob import glob

from prefect import Flow, unmapped
from prefect.executors import DaskExecutor
from cleo import Application, Command

from vss.msft_loader import get_metadata, get_embeddings, batch_data, create_index, load_data


class LoadCommand(Command):
    '''
    Load the VSS data into Redis

    load_msft
        {--redis-url=redis://localhost:6379 : Location of the Redis to Load to}
        {--pipeline-interval=10000 : Amount to break data load into for pipeline}
    '''
    def handle(self):

        metadata_files = glob('data/metadata*')
        pipeline_interval = int(self.option('pipeline-interval'))
        redis_url = self.option('redis-url')
        
        print(metadata_files)

        with Flow('loader', executor=DaskExecutor()) as flow:
            metadata_maps = get_metadata.map(metadata_files)
            embedded_maps = get_embeddings.map(metadata_maps)
            # batched_maps = batch_data(embedded_maps, batch_size)
            load_data.map(*(embedded_maps, unmapped(redis_url), unmapped(pipeline_interval)))

        flow.run()

      
        # embeddings_file = 'data/embeddings_0_0.pkl'
        # self.line(f'<info>Loading Embeddings File:</info> <comment>{embeddings_file}</comment>')
        # start = perf_counter()
        # embeddings = load_embeddings(embeddings_file)
        # end = perf_counter()
        # self.line(f'<comment>{len(embeddings)}</comment> <info>embeddings loaded in</info><comment>{end - start: 0.2f}</comment> <info>seconds</info>')

        # redis_url = self.option('redis-url')
        # self.line(f'<info>Beginning Load into Redis at:</info> <comment>{redis_url}</comment>')
        # r = Redis.from_url(redis_url)
        # vss_index(r, metadata_array[0].keys(), metadata.dtypes, embeddings.shape[1])
        
        # batch_size = self.option('batch-size')
        # start = perf_counter()
        # mass_load_data_to_redis(redis_url, metadata_array, batch_size)

        # batch = 0
        # batch_amount = 10000
        # with r.pipeline(transaction=False) as pipe:
        #     for _metadata, embedding in zip(metadata_array, embeddings):
        #         data = build_object_from_row(_metadata, embedding)
        #         load_vss_obj(pipe, data, offset)
        #         offset += 1
        #         batch += 1
        #         if batch % batch_amount == 0:
        #             pipe.execute()

        #     pipe.execute()

        # end = perf_counter()
        # self.line(f'<info>Loaded data into Redis in</info><comment>{end - start : 0.2f}</comment> <info>seconds</info>')


def run():
    app = Application(name='VSS')
    app.add(LoadCommand())
    app.run()