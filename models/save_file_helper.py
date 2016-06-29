import os

_DEBUGGER_FILES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "debugging_files")


def save_debugging_file(dataframe, file_name):
    if not os.path.exists(_DEBUGGER_FILES_FOLDER):
        os.makedirs(_DEBUGGER_FILES_FOLDER)
    dataframe.to_csv(os.path.join(_DEBUGGER_FILES_FOLDER, file_name), index=False)
