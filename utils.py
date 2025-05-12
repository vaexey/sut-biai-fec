import os

DATA_DIR = "data"

def get_meta():
    class_names = os.listdir(f'{DATA_DIR}/train')
