
FROM  nvcr.io/nvidia/pytorch:22.07-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY core_requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]