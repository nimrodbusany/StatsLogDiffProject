import os


def create_folder_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)