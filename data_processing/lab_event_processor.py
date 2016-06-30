from data_loading.data_loaders import get_lab_events
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.event_processor import resample_flatten_and_add_diff_values_to_events
from data_processing.processed_data_interface import cache_results

_REGULAR_LAB_ITEM_FIELDS = [
    'creat',
    'hct',
    'hgb',
    'potassium',
    'sodium',
    'urea_n'
]

ALL_LAB_ITEM_FIELDS = _REGULAR_LAB_ITEM_FIELDS + [field + "_diff" for field in _REGULAR_LAB_ITEM_FIELDS]

@cache_results("processed_lab_items.csv")
def get_processed_lab_events(use_cache=True):
    """Returns a dataframe where each row a unique combination of date and icustay_id and has a column for each lab
    item and lab item diff for each different lab item that is used in the analysis. The icustay_id and date columns
    are used to join the records with other post processed data.

    Args:
        use_cache: Whether to load the results from a previous calculation.

    Returns:
        A dataframe with the following columns:
        icustay_id:
        date:
        alt(sgpt):
        alt(sgpt)_diff:
        ast(sgot):
        ast(sgot)_diff:
        creat:
        creat_diff:
        ctropni:
        ctropni_diff:
        ctropnt:
        ctropnt_diff:
        hct:
        hct_diff:
        heinz:
        heinz_diff:
        hgb:
        hgb_diff:
        potassium:
        potassium_diff:
        probnp:
        probnp_diff:
        sodium:
        sodium_diff:
        urea_n:
        urea_n_diff:
        uric_acid:
        uric_acid_diff:

    """
    lab_events = get_lab_events()
    modify_dates_fn = get_modify_dates_fn()

    lab_events.drop('value', axis=1, inplace=True)
    lab_events.rename(columns={"valuenum": "value"}, inplace=True)
    lab_events = modify_dates_fn(lab_events, ["charttime"])
    fields_to_keep = ['icustay_id', 'date'] + ALL_LAB_ITEM_FIELDS

    return resample_flatten_and_add_diff_values_to_events(lab_events)[fields_to_keep]
