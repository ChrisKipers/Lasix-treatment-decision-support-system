import os
import shutil
import logging
import time

import pandas as pd

_PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data")

_DATE_FIELDS = ['charttime', 'date']


def cache_results(file_name, description):
    """Decorator used by data processing methods to cache results to csv files

    Args:
        file_name: The file name to use as the cache file.

    Returns:
        A decorator function for the given file_name.
    """

    def decorate(func):

        def cacher(**karg):
            use_cache = karg["use_cache"] if "use_cache" in karg else True
            if _does_file_exist(file_name) and use_cache:
                logging.info("Loading %s from cache" % description)
                start = time.perf_counter()
                data = _load_data_frame(file_name)
                for date_field in _DATE_FIELDS:
                    if date_field in data.columns:
                        data[date_field] = pd.to_datetime(data[date_field])
                stop = time.perf_counter()
                logging.info("Loading %s from cache took %.3fs" % (description, stop - start))
            else:
                logging.info("Processing %s" % description)
                start = time.perf_counter()
                data = func(**karg)
                stop = time.perf_counter()
                logging.info("Processing %s took %.3fs" % (description, stop - start))
                _save_data_frame(data, file_name)
            return data

        return cacher

    return decorate

def clear_processed_data_cache():
    """Removes all cached preprocessed data"""
    if os.path.exists(_PROCESSED_DATA_DIR):
        shutil.rmtree(_PROCESSED_DATA_DIR)

def _does_file_exist(file_name):
    return os.path.exists(os.path.join(_PROCESSED_DATA_DIR, file_name))


def _save_data_frame(dataframe, file_name):
    if not os.path.exists(_PROCESSED_DATA_DIR):
        os.makedirs(_PROCESSED_DATA_DIR)
    dataframe.to_csv(os.path.join(_PROCESSED_DATA_DIR, file_name), index=False)


def _load_data_frame(file_name):
    return pd.read_csv(os.path.join(_PROCESSED_DATA_DIR, file_name))
