import pandas as pd
from data_processing.chart_event_processor import get_processed_chart_events
from data_processing.lab_event_processor import get_processed_lab_events
from data_processing.get_patient_info import get_patient_info
from data_processing.hospital_admission_outcome import get_outcome
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
    lab_events = get_processed_lab_events(use_cache=use_cache)
    chart_events = get_processed_chart_events(use_cache=use_cache)
    patients = get_patient_info(use_cache=use_cache)
    lasix = get_processed_lasix(use_cache=use_cache)

    current_merged_df = pd.merge(
        lasix,
        lab_events,
        how="inner"  # Keep rows even if there is not a row from each df
    )

    current_merged_df = pd.merge(
        current_merged_df,
        chart_events,
        how="inner"  # Keep rows even if there is not a row from each df
    )

    current_merged_df = pd.merge(
        current_merged_df,
        patients,
        how="inner"
    )

    outcome_df = get_outcome(current_merged_df, death_time_frame=3)
    current_merged_df = pd.merge(
        current_merged_df,
        outcome_df,
        how="inner"
    )

    return current_merged_df

