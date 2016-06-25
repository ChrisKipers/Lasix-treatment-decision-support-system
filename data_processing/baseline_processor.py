import pandas as pd
from pandas.tseries.offsets import Day

from data_loading.data_loaders import get_lab_events, get_lab_item_details
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.processed_data_interface import cache_results

@cache_results("baseline_labevents.csv")
def process_labevents(use_cache=False):
    """Processes lab events into a format that can be merged with dataframes with [hadm_id, date] primary key.
    A row in the lab events dataframe is a specific lab item and value for that item at a specific time. A row in
    the output dataframe is all values for every lab item for a hadm_id and day.

    The transformation process has two critical steps:

    1. The lab item events need to be resampled with a constant frequency. There can be many lab items for a single day
    and some days where there are no lab items. We need to modify the data so that there is only one lab item for
    each day. This is done by resampling the lab events to a daily frequency, picking the first lab item for the day
    if there are multiple lab items a day. After resampling, a forward filling (ffill) process is performed to fill
    in any missing lab items for a day with the previous days value. Before this transformation takes place, the lab
    items are grouped by hadm_id and itemid so that the resampling and filling is unique per item type and hadm.

    2. In the input dataframe, each row only has one lab item, with a itemid column to differentiate the lab item type.
    In the output dataframe, each row needs to have a column for every item type, and the value of the columns are the
    values for that item type for a specific hadm_id and day. We perform this transformation with padas pivot_table
    function.

    Args:
        use_cache: Skip computation and load results from previous computation


    Returns:
        A DataFrame with the following columns:
        hadm_id: The hospital admission ID for the lab results.
        charttime: The day the lab results were collected.
        alt(sgpt):
        ast(sgot):
        creat:
        ctropni:
        ctropnt:
        hct:
        hgb:
        potassium:
        probnp:
        sodium:
        urea_n:
        uric_acid:
    TODO enter comments for each column, including units
    """
    lab_events = get_lab_events()
    target_lab_item_ids = [
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

    # Cleanup values
    common_invalid_value_prefixes = ["<", ">", "LESS THAN ", "GREATER THAN"]
    for prefix in common_invalid_value_prefixes:
        lab_events.value = lab_events.value.str.replace(prefix, "")

    lab_events.value = pd.to_numeric(lab_events.value, errors="coerce")
    lab_events.dropna(inplace=True, subset=["value"])

    target_lab_events = lab_events[lab_events.itemid.isin(target_lab_item_ids)]

    lab_item_details = get_lab_item_details()
    lab_item_details.TEST_NAME = lab_item_details.TEST_NAME.str.replace(" ", "_").str.lower()

    lab_item_name_by_id = {t.ITEMID: t.TEST_NAME for t in lab_item_details.itertuples()}

    target_lab_events["test"] = target_lab_events.itemid.map(lab_item_name_by_id)

    modify_dates_fn = get_modify_dates_fn()
    target_lab_events = modify_dates_fn(target_lab_events, ["charttime"])
    target_lab_events = target_lab_events.set_index(target_lab_events.charttime)
    # TODO modify value based on valueuom. I making a stuipd assumption that all the valueuom are the same
    # for each item type.
    # TODO verify values are float, I saw some strings like "<3.0"
    target_lab_event_fields = target_lab_events[["hadm_id", "test", "value", "charttime"]]

    lab_events_grouped_by_hadm_and_test = target_lab_event_fields.groupby(["hadm_id", "test"])
    # TODO find out if values for the same day are the values at the beggining for end of day
    lab_events_grouped_by_hadm_and_test_filled = lab_events_grouped_by_hadm_and_test.resample("D").first().ffill()

    # Save charttime to another column so that it is not lost when dropping the indexes
    lab_events_grouped_by_hadm_and_test_filled.charttime = \
        lab_events_grouped_by_hadm_and_test_filled.index.get_level_values("charttime")
    # Drop grouping
    lab_events_grouped_by_hadm_and_test_filled.reset_index(drop=True, inplace=True)

    # Reshape DF so that each row is unique for a hadm_id and day and each test type is a column
    pivot = pd.pivot_table(lab_events_grouped_by_hadm_and_test_filled,
                           values="value", columns=["test"], index=["hadm_id", "charttime"])

    # TODO change charttime to date to be consistent
    # Flatten out the index so that that they are columns
    return pivot.reset_index()

@cache_results("lab_results_with_previous_day_results.csv")
def get_lab_results_with_previous_day_results(use_cache=False):
    lab_results = process_labevents()
    previous_day_lab_results = lab_results.copy()
    previous_day_lab_results.charttime = previous_day_lab_results.charttime - Day(1)

    merged_results = pd.merge(
        lab_results,
        previous_day_lab_results,
        on=["hadm_id", "charttime"],
        how="left",
        suffixes=('', '_previous'))

    # For each previous column, if the value is null, we will set the previous value to the current value.
    # In reality only the first day for each admission should have missing previous values, but this approach is very
    # clean and should not produce negative side effects.

    # previous_columns = merged_results.columns[merged_results.columns.str.contains('_previous')]
    # normal_columns = previous_columns.str.replace('_previous', '')
    # for (column, previous_column) in zip(normal_columns, previous_columns):
    #     missing_previous_column = merged_results[previous_column].isnull()
    #     merged_results[missing_previous_column][previous_column] =\
    #         merged_results[missing_previous_column][column]

    return merged_results
