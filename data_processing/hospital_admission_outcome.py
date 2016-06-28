import pandas as pd

from pandas.tseries.offsets import Day

from data_loading.data_loaders import get_patients
from data_processing.processed_data_interface import cache_results

@cache_results("hospital_admission_outcome.csv")
def get_hospital_admission_outcomes(use_cache=False):
    """Returns a dataframe containing whether a patient died or not during a hospital visit.

    Args:
        use_cache: Skip computation and load results from previous computation.

    Returns:
        A DataFrame with the following columns:
        hadm_id: The ID of the hospital admission.
        died: Whether the patient died during the hospital visit.
    """
    patients = get_patients()
    died = patients.dod.isnull()

    return pd.DataFrame({"hadm_id": patients.hadm_id, "died": died})

def get_outcome(data, death_time_frame):
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
