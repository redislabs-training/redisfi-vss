FROM python:3.9

RUN pip3 install --upgrade pip
RUN pip3 install poetry

ENV INSTALL_DIR=/opt/redisfi-vss
WORKDIR ${INSTALL_DIR}
COPY pyproject.toml .

RUN poetry install 

# Development workaround - make sure we don't override poetry.lock inside the container with one we're copying from local
RUN mv poetry.lock poetry.lock.bak
COPY . .
RUN mv poetry.lock.bak poetry.lock

# Installs the package in the environment
RUN poetry install

## downloads model dependancies for ML libs
RUN poetry run VSS
ENV PYTHONUNBUFFERED=1


