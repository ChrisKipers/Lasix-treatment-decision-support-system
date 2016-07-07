import os
import shutil

import pandas as pd

_DEBUGGER_FILES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "debugging_files")


def save_debugging_file(dataframe, file_name):
    if not os.path.exists(_DEBUGGER_FILES_FOLDER):
        os.makedirs(_DEBUGGER_FILES_FOLDER)
    dataframe.to_csv(os.path.join(_DEBUGGER_FILES_FOLDER, file_name), index=False)

def get_debugging_file(file_name):
    return pd.read_csv(os.path.join(_DEBUGGER_FILES_FOLDER, file_name))

def delete_model_debugging_files():
    if os.path.exists(_DEBUGGER_FILES_FOLDER):
        shutil.rmtree(_DEBUGGER_FILES_FOLDER)


