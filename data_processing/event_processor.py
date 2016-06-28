import pandas as pd
from pandas.tseries.offsets import Day

def process_events(event_records):
    """Processes lab events into a format that can be merged with dataframes with [icustay_id, date] primary key.
    A row in the lab events dataframe is a specific lab item and value for that item at a specific time. A row in
    the output dataframe is all values for every lab item for a icustay_id and day.

    The transformation process has two critical steps:

    1. The lab item events need to be resampled with a constant frequency. There can be many lab items for a single day
    and some days where there are no lab items. We need to modify the data so that there is only one lab item for
    each day. This is done by resampling the lab events to a daily frequency, picking the first lab item for the day
    if there are multiple lab items a day. After resampling, a forward filling (ffill) process is performed to fill
    in any missing lab items for a day with the previous days value. Before this transformation takes place, the lab
    items are grouped by icustay_id and itemid so that the resampling and filling is unique per item type and hadm.

    2. In the input dataframe, each row only has one lab item, with a itemid column to differentiate the lab item type.
    In the output dataframe, each row needs to have a column for every item type, and the value of the columns are the
    values for that item type for a specific icustay_id and day. We perform this transformation with padas pivot_table
    function.

    Args:
        use_cache: Skip computation and load results from previous computation


    Returns:
        A DataFrame with the following columns:
        icustay_id: The hospital admission ID for the lab results.
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
    event_records.label = event_records.label.str.replace(" ", "_").str.lower()

    target_events = event_records.set_index(event_records.charttime)

    target_event_fields = target_events[["icustay_id", "label", "value", "charttime"]]

    events_grouped_by_hadm_and_label = target_event_fields.groupby(["icustay_id", "label"])

    events_grouped_by_hadm_and_label_filled = events_grouped_by_hadm_and_label.resample("D").first().ffill()

    # Save charttime to another column so that it is not lost when dropping the indexes
    events_grouped_by_hadm_and_label_filled.charttime = \
        events_grouped_by_hadm_and_label_filled.index.get_level_values("charttime")
    # Drop grouping
    events_grouped_by_hadm_and_label_filled.reset_index(drop=True, inplace=True)

    # Reshape DF so that each row is unique for a icustay_id and day and each label type is a column
    pivot = pd.pivot_table(events_grouped_by_hadm_and_label_filled,
                           values="value", columns=["label"], index=["icustay_id", "charttime"])

    # TODO change charttime to date to be consistent
    # Flatten out the index so that that they are columns
    return pivot.reset_index().rename(columns={"charttime": "date"})

def add_event_value_diffs_to_flattend_events(flatten_events):

    columns_to_get_diffs = [column for column in flatten_events.columns if column not in ["icustay_id", "date"]]

    previous_day_even_results = flatten_events.copy()
    previous_day_even_results.date = previous_day_even_results.date - Day(1)

    merged_results = pd.merge(
        flatten_events,
        previous_day_even_results,
        on=["icustay_id", "date"],
        how="left",
        suffixes=('', '_previous'))

    for column in columns_to_get_diffs:
        current = merged_results[column]
        previous = merged_results[column + "_previous"]
        diff = current - previous
        merged_results[column + "_diff"] = diff

    columns_to_drop = [column + "_previous" for column in columns_to_get_diffs]
    return merged_results.drop(columns_to_drop, axis=1)
