from data_loading.data_loaders import get_chart_events
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.event_processor import resample_flatten_and_add_diff_values_to_events
from data_processing.processed_data_interface import cache_results

_REGULAR_CHART_ITEM_FIELDS = [
    'glucose_(70-105)',
    'heart_rate',
    'hematocrit',
    'hemoglobin',
    'magnesium_(1.6-2.6)',
    'potassium_(3.5-5.3)',
    'respiratory_rate',
    'spo2',
    'temperature_c_(calc)'
]

ALL_CHART_ITEM_FIELDS = _REGULAR_CHART_ITEM_FIELDS + [field + "_diff" for field in _REGULAR_CHART_ITEM_FIELDS]

@cache_results("processed_chart_events.csv")
def get_processed_chart_events(use_cache=True):
    """Returns a dataframe where each row a unique combination of date and icustay_id and has a column for each chart
    item and chart item diff for each different chart item that is used in the analysis. The icustay_id and date columns
    are used to join the records with other post processed data.

    Args:
        use_cache: Whether to load the results from a previous calculation.

    Returns:
        A dataframe with the following columns:
        icustay_id:
        date,
        glucose_(70-105):
        glucose_(70-105)_diff:
        heart_rate:
        heart_rate_diff:
        hematocrit:
        hematocrit_diff:
        hemoglobin:
        hemoglobin_diff:
        magnesium_(1.6-2.6):
        magnesium_(1.6-2.6)_diff:
        potassium_(3.5-5.3):
        potassium_(3.5-5.3)_diff:
        respiratory_rate:
        respiratory_rate_diff:
        spo2:
        spo2_diff:
        temperature_c_(calc):
        temperature_c_(calc)_diff:
    """
    modify_dates_fn = get_modify_dates_fn()
    chart_events = get_chart_events()

    chart_events.rename(columns={"value1num": "value"}, inplace=True)
    chart_events = chart_events[['subject_id', 'icustay_id', 'charttime', 'itemid', 'label', 'value']]

    chart_events = modify_dates_fn(chart_events, ['charttime'])
    # Modify shape of dataframe so that each chart item has its own column.
    fields_to_keep = ['icustay_id', 'date'] + ALL_CHART_ITEM_FIELDS
    return resample_flatten_and_add_diff_values_to_events(chart_events)[fields_to_keep]
