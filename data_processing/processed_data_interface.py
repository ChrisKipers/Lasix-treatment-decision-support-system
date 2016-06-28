import pandas as pd
import os

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data")

_DATE_FIELDS = ['charttime', 'charttime', 'date']

def does_file_exist(file_name):
    return os.path.exists(os.path.join(PROCESSED_DATA_DIR, file_name))


def save_data_frame(dataframe, file_name):
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    dataframe.to_csv(os.path.join(PROCESSED_DATA_DIR, file_name), index=False)


def load_data_frame(file_name):
    return pd.read_csv(os.path.join(PROCESSED_DATA_DIR, file_name))

def cache_results(file_name):
    def decorate(func):

        def cacher(**karg):
            use_cache = karg["use_cache"] if "use_cache" in karg else True
            if does_file_exist(file_name) and use_cache:
                data = load_data_frame(file_name)
                for date_field in _DATE_FIELDS:
                    if date_field in data.columns:
                        data[date_field] = pd.to_datetime(data[date_field])
            else:
                data = func(**karg)
                save_data_frame(data, file_name)
            return data

        return cacher

    return decorate
