
FROM python:3.9.1-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# install git    
RUN apt-get -y update
RUN apt-get -y install git

#download repo
RUN git clone https://github.com/JC-me-coding/Project_MLOps/ /Project_MLOps


#COPY core_requirements.txt requirements.txt
#COPY setup.py setup.py
#COPY src/ src/
WORKDIR /
RUN pip install -r /Project_MLOps/requirements.txt --no-cache-dir

#COPY dummy.py dummy.py
ENTRYPOINT ["python", "-u", "Project_MLOps/dummy.py"]
