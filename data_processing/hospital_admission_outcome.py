import pandas as pd

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
