# https://github.com/ykursadkaya/pyspark-Docker
ARG IMAGE_VARIANT=buster
ARG OPENJDK_VERSION=11
ARG PYTHON_VERSION=3.9.8

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.2.2
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}

RUN apt update -y && apt install git

# install requirements
COPY ./requirements.txt ./
COPY ./requirements-dev.txt ./
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt
