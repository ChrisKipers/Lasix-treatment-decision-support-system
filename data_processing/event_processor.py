import pandas as pd
from pandas.tseries.offsets import Day


def resample_flatten_and_add_diff_values_to_events(event_records):
    """Reformats event data so that it can be used by machine learning models. Raw event data is structured so that each
    row represents a single event item. The event item contains a type, a value, icustay_id and charttime. The
    transformation processes reorganizes this data so that each row represents a single day in the icu with each
    different event type getting its own column and column containing how the value changed from the previous day.

    Note:
        See documentation in other methods for more information on the different transformation steps.

    Args:
        A dataframe containing the following columns:
        icustay_id: The icustay the event was recorded for.
        label: The type of the event.
        value: The value for the event.
        charttime: When the event was recorded.

    Returns
        A dataframe with the following columns
        icustay_id: The icustay the event was recorded for.
        date: The day the event was recorded.
        label1: The value for event with label 1.
        label1_diff: The value difference for label 1 from the previous day to the current day.
        ...
        labeln: The value for event with label n.
        labeln_diff: The value difference for label n from the previous day to the current day.
    """
    resampled_event_data = _resample_and_fill_event_items_grouped_by_icustay(event_records)
    flattened_event_records = _flatten_events(resampled_event_data)
    return _add_event_value_diffs_to_flattened_events(flattened_event_records)


def _resample_and_fill_event_items_grouped_by_icustay(event_records):
    """Resamples and forward fills events grouped by label and icustay_id. The purpose of this transformation is to
    create a normalized view into how the patients event values change day by day.

    Example:
        Input:
        icustay_id  | label | value | charttime
        1           | HR    | 60    | 6/2/16 5:00
        1           | HR    | 55    | 6/2/16 6:30
        1           | HR    | 75    | 6/2/16 7:32
        1           | HR    | 55    | 6/4/16 8:00

        Output:
        icustay_id  | label | value | date
        1           | HR    | 60    | 6/2/16
        1           | HR    | 75    | 6/3/16
        1           | HR    | 55    | 6/4/16

        In this example, the output contains 3 columns, for each date that values were observed for. The value for
        6/3/16 was forward filled from the last recorded value on 6/2/16

    Args:
        A dataframe containing the following columns:
        icustay_id: The icustay the event was recorded for.
        label: The type of the event.
        value: The value for the event.
        charttime: When the event was recorded.

    Returns:
        A dataframe with the following columns:
        icustay_id: The icustay the event was recorded for.
        label: The type of the event.
        value: The value for the event.
        date: The day the event was recorded.
    """
    target_events = event_records.set_index(event_records.charttime)

    target_event_fields = target_events[["icustay_id", "label", "value", "charttime"]]

    events_grouped_by_hadm_and_label = target_event_fields.groupby(["icustay_id", "label"])

    events_grouped_by_hadm_and_label_filled = events_grouped_by_hadm_and_label.resample("D").first().ffill()

    # Save charttime to another column so that it is not lost when dropping the indexes
    events_grouped_by_hadm_and_label_filled.charttime = \
        events_grouped_by_hadm_and_label_filled.index.get_level_values("charttime")
    # Drop grouping
    return events_grouped_by_hadm_and_label_filled.reset_index(drop=True)


def _flatten_events(event_records):
    """Flattens the events so that each row is unique for icustay_id and date and each different type of event item
    has its own columns. The purpose of this transformation is to provide a record which represents all event items for
    a given icustay on a given day.

    Args:
        A dataframe with the following columns:
        icustay_id: The icustay the event was recorded for.
        date: The day the event was recorded.
        label: The type of the event.
        value: The value for the event.

    Returns:
        A dataframe with the following columns
        icustay_id: The icustay the event was recorded for.
        date: The day the event was recorded.
        label1: The value for event with label 1.
        ...
        labeln: The value for event with label n.

    """
    event_records.label = event_records.label.str.replace(" ", "_").str.lower()

    # Reshape DF so that each row is unique for a icustay_id and day and each label type is a column
    pivot = pd.pivot_table(event_records, values="value", columns=["label"], index=["icustay_id", "charttime"])

    # Flatten out the index so that that they are columns
    return pivot.reset_index().rename(columns={"charttime": "date"})


def _add_event_value_diffs_to_flattened_events(flattened_events):
    """Adds the event value diff from the previous day to the current day for each event type.

    Example:
        input:
        icustay_id  | date      |   label1  | label2
        1           | 6/2/16    |   1       | 6
        1           | 6/3/16    |   4       | 5
        1           | 6/4/1     |   2       | 8

        output:
        icustay_id  | date      |   label1  | label1_diff   | label2    | label2_diff
        1           | 6/2/16    |   1       | 0             | 6         | 0
        1           | 6/3/16    |   4       | 3             | 5         | -1
        1           | 6/4/1     |   2       | -2            | 8         | 3

    Note:
        The first diffs for each event type for the first day in an icustay will be 0 since there is no previous value
        to compare it to.

    Args:
        A dataframe with the following columns
        icustay_id: The icustay the event was recorded for.
        date: The day the event was recorded.
        label1: The value for event with label 1.
        ...
        labeln: The value for event with label n.

    Returns
        A dataframe with the following columns
        icustay_id: The icustay the event was recorded for.
        date: The day the event was recorded.
        label1: The value for event with label 1.
        label1_diff: The value difference for label 1 from the previous day to the current day.
        ...
        labeln: The value for event with label n.
        labeln_diff: The value difference for label n from the previous day to the current day.
    """

    columns_to_get_diffs = [column for column in flattened_events.columns if column not in ["icustay_id", "date"]]

    previous_day_even_results = flattened_events.copy()
    previous_day_even_results.date = previous_day_even_results.date - Day(1)

    merged_results = pd.merge(
        flattened_events,
        previous_day_even_results,
        on=["icustay_id", "date"],
        how="left",
        suffixes=('', '_previous'))

    for column in columns_to_get_diffs:
        current = merged_results[column]
        previous = merged_results[column + "_previous"]
        diff = current - previous
        # If a diff is na, assume that the value didn't change since yesterday.
        diff.fillna(0, inplace=True)
        merged_results[column + "_diff"] = diff

    columns_to_drop = [column + "_previous" for column in columns_to_get_diffs]
    return merged_results.drop(columns_to_drop, axis=1)
