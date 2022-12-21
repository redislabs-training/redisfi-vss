# redisfi-vss

RedisFI VSS is a data loader/microservice to facilitate the VSS part of [RedisFI](https://github.com/redislabs-training/redisfi).  It is a search engine on top of a collection of SEC 10K-Q documents that were vectorized by Microsoft's Enterprise Data Science team (who has graciously allowed us to share them with this project) using [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).

## Prereqs
Both forms require [Python 3.9](https://stackoverflow.com/a/66907362) (not above) and [Poetry](https://python-poetry.org)

## Setup 

Once Poetry is installed, setting up redisfi-vss is straight forward.  Inside the directory run:

`poetry install`

Once the dependancies are installed, you should be able to run the CLI by typing:

`poetry run VSS`

Which will print a menu of the different options available.  The main two are `load` - which will load the prepared data into Redis Enterprise - and `run`, which will run the microservice to make searching the data possible.

The data set is very large and requires a Redis Cluster to fully work. For testing, it's recommended to only load one of the data files.