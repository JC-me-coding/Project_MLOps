import zipfile
import os
from dvc.repo import Repo
import glob


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
    # Is it a git repo? Check if .github folder exists
    root = os.getcwd()
    if os.path.exists(".github"):
        if len(dvc_status()) != 0:
            print("Data is not up to date, pulling data")
            dvc_pull()
        else:
            print("Data is up to date")
        
        # check if extracted data already exists
        if not os.path.exists(f"{root}/data/processed/landscapes"):
            extract_zip(f"{root}/data/raw/landscapes.zip", f"{root}/data/processed/")
    # Are we on a GCP instance, where buckets is mounted
    elif os.path.exists("/gcs"):
        os.makedirs(f"{root}/data/processed/", exist_ok=True)
        dvc_folder = "/gcs/landscapes-team19"
        list_of_files = glob.glob(f'{dvc_folder}/*/*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        extract_zip(latest_file, f"{root}/data/processed/")