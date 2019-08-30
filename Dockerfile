FROM python:3.7
MAINTAINER AFC "alexisfcote@gmail.com"
COPY requirements.txt .
RUN pip install pipenv

# -- Adding Pipfiles
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN pipenv install --deploy --system