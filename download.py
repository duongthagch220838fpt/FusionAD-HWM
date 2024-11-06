import subprocess
import os
import shutil
import argparse
import gdown

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="medicine_pack",
                    type=str, help="dataset classname")
args = vars(parser.parse_args())


def install_package(package):
    subprocess.check_call(["pip", "install", package])


def download_file(file_id, output_name):
    install_package("gdown")
    import gdown
    gdown.download(id=file_id, output=output_name, quiet=False)


def unzip_file(zip_file):
    if not os.path.exists("data"):
        os.mkdir("data")
    import zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("data")


def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


download_file('1TmZIQTEaL5m4IFBs4NiF0Wj-1POaaGJ0', 'medicine_pack.zip')
unzip_file('medicine_pack.zip')
