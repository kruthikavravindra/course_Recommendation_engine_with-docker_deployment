FROM ubuntu:latest

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install pip -y
RUN pip install joblib
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit_learn
RUN pip install pickle-mixin

COPY train.csv ./train.csv
COPY test.csv ./test.csv
COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py
RUN python3 inference.py