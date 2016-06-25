import pandas as pd

from data_processing.baseline_processor import get_lab_results_with_previous_day_results
from data_processing.get_patient_info import get_patient_info
from data_processing.hospital_admission_outcome import get_hospital_admission_outcomes
from data_processing.lasix_poe_processor import get_processed_lasix
from data_processing.processed_data_interface import cache_results

@cache_results("ml_data.csv")
def get_ml_data(use_cache=False):
    """Returns a dataframe that can be used to train a ML model to predict the outcome of a congestive
    heart failure patient.

    Args:
        use_cache: Skip computation and load results from previous computation.

    Returns:
        A DataFrame with the following columns:
        date:
        hadm_id:
        subject_id:
        treatment:
        charttime:
        alt(sgpt):
        ast(sgot):
        ctropnt:
        hct:
        hgb:
        potassium:
        probnp:
        sodium:
        urea_n:
        uric_acid:
        sex:
        marital_status_descr:
        ethnicity_descr:
        overall_payor_group_descr:
        religion_descr:
        age:
        died:
    """
    lab_events = get_lab_results_with_previous_day_results(use_cache=use_cache)
    patients = get_patient_info(use_cache=use_cache)
    hospital_outcomes = get_hospital_admission_outcomes(use_cache=use_cache)
    lasix = get_processed_lasix(use_cache=use_cache)

    current_merged_df = pd.merge(
        lasix,
        lab_events,
        left_on=["hadm_id", "date"],
        right_on=["hadm_id", "charttime"],
        how="outer"  # Keep rows even if there is not a row from each df
    )

    current_merged_df = pd.merge(
        current_merged_df,
        patients,
        on=["subject_id", "hadm_id"]
    )

    current_merged_df = pd.merge(
        current_merged_df,
        hospital_outcomes,
        on=["hadm_id"]
    )

    return current_merged_df

