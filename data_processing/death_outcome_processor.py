import pandas as pd
from pandas.tseries.offsets import Day

from data_loading.data_loaders import get_patients


def get_death_outcome(data, death_time_frame):
    """Creates a data frame that contains whether the patient died within a time frame.

    Args:
        data: A dataframe containing the following columns:
        subject_id: The ID of the subject.
        date: The date to compare against the date of death.

        death_time_frame: The number of days that a patient must be alive after for the died outcome to be false.

    Returns:
        A dataframe with the following columns:
        subject_id: The ID of the subject.
        date: The date to compare against the date of death.
        died: Whether the patient died within the time frame from the date.
    """
    patients = get_patients()
    dod_by_subject_id = {p.subject_id: p.dod for p in patients.itertuples()}
    days_unit = Day(death_time_frame)
    dod = data.subject_id.map(dod_by_subject_id)
    end_of_type_frame = data.date.map(lambda d: d + days_unit)
    date_out_of_range = end_of_type_frame.max() + days_unit
    died_within_time = end_of_type_frame >= dod.fillna(date_out_of_range).astype(end_of_type_frame.dtype)
    outcome_df = pd.concat([data.subject_id, data.date, died_within_time], axis=1)
    outcome_df.columns = ['subject_id', 'date', 'died']
    return outcome_df
