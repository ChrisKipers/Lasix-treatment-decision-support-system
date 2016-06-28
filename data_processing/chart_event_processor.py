from data_loading.data_loaders import get_chart_events
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.event_processor import process_events, add_event_value_diffs_to_flattend_events
from data_processing.processed_data_interface import cache_results

__TARGET_CHART_EVENTS = [
    211, # heart rate
    813, # hematrocrit
    814, # hemoglobin
    821, # magnesium
    827, # phosphorous
    829, # potassium
    618, # respiratory rate
    646, # spo2
    677, # temperature
    811, # glucose
    762 # admit_weight
]

@cache_results("processed_chart_events.csv")
def get_processed_chart_events(use_cache=True):
    modify_dates_fn = get_modify_dates_fn()
    chart_events = get_chart_events()

    target_chart_events = chart_events[chart_events.itemid.isin(__TARGET_CHART_EVENTS)]
    target_chart_events.rename(columns={"value1num": "value"}, inplace=True)
    target_chart_events = target_chart_events[['subject_id', 'icustay_id', 'charttime', 'itemid', 'label', 'value']]

    target_chart_events = modify_dates_fn(target_chart_events, ['charttime'])

    flattened_events = process_events(target_chart_events)
    return add_event_value_diffs_to_flattend_events(flattened_events)
