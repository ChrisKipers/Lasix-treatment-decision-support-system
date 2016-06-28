from data_loading.data_loaders import get_lab_events
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.event_processor import process_events, add_event_value_diffs_to_flattend_events
from data_processing.processed_data_interface import cache_results


__TARGET_LAB_ITEM_IDS = [
    50159, # Sodium in Blood
    50090, # Creatine
    50177, # Urea nitrogen
    50195, # BNP
    50073, # AST
    50062, # ALT
    50188, # Troponin I
    50189, # Troponin T
    50384, # Hematocrit
    50386, # Hemoglobin
    50277, # Sodium in urine
    50178, # Uric acid in blood
    50149, # Potassium
    50383, # Hematocrit
]

@cache_results("processed_lab_items.csv")
def get_processed_lab_events(use_cache=True):
    lab_events = get_lab_events()
    modify_dates_fn = get_modify_dates_fn()

    target_lab_events = lab_events[lab_events.itemid.isin(__TARGET_LAB_ITEM_IDS)]
    target_lab_events.drop('value', axis=1, inplace=True)
    target_lab_events.rename(columns={"valuenum": "value"}, inplace=True)
    target_lab_events = modify_dates_fn(target_lab_events, ["charttime"])

    flattened_events = process_events(target_lab_events)
    return add_event_value_diffs_to_flattend_events(flattened_events)
