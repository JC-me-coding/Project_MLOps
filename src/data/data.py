import zipfile
import os
from dvc.repo import Repo


def extract_zip(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    with zipfile.ZipFile(input_filepath, "r") as zip_ref:
        zip_ref.extractall(output_filepath)


def dvc_pull():
    repo = Repo(".")
    repo.pull()

def dvc_status():
    repo = Repo(".")
    return repo.status()


if __name__ == "__main__":
    #print(dvc_status())
    root = os.getcwd()
    if len(dvc_status()) != 0:
        print("Data is not up to date, pulling data")
        dvc_pull()
        extract_zip(f"{root}/data/raw/landscapes.zip", f"{root}/data/processed/")
    else:
        print("Data is up to date")


