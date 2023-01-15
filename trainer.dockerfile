
FROM python:3.9.1-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY core_requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY setup.py setup.py
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY Makefile Makefile

WORKDIR /

ENTRYPOINT ["scripts/trainer.sh"]
