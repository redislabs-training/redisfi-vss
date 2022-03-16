from time import perf_counter

from cleo import Application, Command
from redis import Redis

from vss.db import create_index, load_vss_obj
from vss.msft_loader import load_metadata, munge_metadata, load_embeddings, build_object_from_row


class LoadCommand(Command):
    '''
    Load the VSS data into Redis

    load_msft
    '''
    def handle(self):
        metadata_file = 'data/metadata_0_0.parquet'
        self.line(f'<info>Loading Metadata File:</info> <comment>{metadata_file}</comment>')
        start = perf_counter()
        metadata = load_metadata(metadata_file)
        metadata = munge_metadata(metadata)
        offset = metadata.index.start 
        metadata_array = metadata.to_dict('records')
        end = perf_counter()
        self.line(f'<comment>{len(metadata_array)}</comment> <info>metadata records prepared in</info><comment>{end - start: 0.2f}</comment> <info>seconds</info>')
        
        embeddings_file = 'data/embeddings_0_0.pkl'
        self.line(f'<info>Loading Embeddings File:</info> <comment>{embeddings_file}</comment>')
        start = perf_counter()
        embeddings = load_embeddings(embeddings_file)
        end = perf_counter()
        self.line(f'<comment>{len(embeddings)}</comment> <info>embeddings loaded in</info><comment>{end - start: 0.2f}</comment> <info>seconds</info>')

        start = perf_counter()
        self.line('<info>Beginning Load into Redis</info>')
        r = Redis()
        create_index(r, [])

        batch = 0
        batch_amount = 10000
        with r.pipeline(transaction=False) as pipe:
            for _metadata, embedding in zip(metadata_array, embeddings):
                data = build_object_from_row(_metadata, embedding)
                load_vss_obj(pipe, data, offset)
                offset += 1
                batch += 1
                if batch % batch_amount == 0:
                    pipe.execute()

            pipe.execute()

            
        end = perf_counter()
        self.line(f'<info>Loaded data into Redis in</info><comment>{end - start : 0.2f}</comment> <info>seconds</info>')


def run():
    app = Application(name='VSS')
    app.add(LoadCommand())
    app.run()