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
    print("dvc pull to get latest data...")
    repo = Repo(".")
    repo.pull()


if __name__ == "__main__":
    dvc_pull()
    root = os.getcwd()
    extract_zip(f"{root}/data/raw/landscapes.zip", f"{root}/data/processed/")