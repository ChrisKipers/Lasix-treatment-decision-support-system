import pandas as pd

from data_processing.chart_event_processor import get_processed_chart_events
from data_processing.lab_event_processor import get_processed_lab_events
from data_processing.patient_info_processor import get_processed_patient_info
from data_processing.death_outcome_processor import get_death_outcome
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
        icustay_id:
        treatment:
        sex:
        marital_status_descr:
        ethnicity_descr:
        overall_payor_group_descr:
        religion_descr:
        age:
        died:
        alt(sgpt):
        alt(sgpt)_diff:
        ast(sgot):
        ast(sgot)_diff:
        creat:
        creat_diff:
        ctropni:
        ctropni_diff:
        ctropnt:
        ctropnt_diff:
        hct:
        hct_diff:
        heinz:
        heinz_diff:
        hgb:
        hgb_diff:
        potassium:
        potassium_diff:
        probnp:
        probnp_diff:
        sodium:
        sodium_diff:
        urea_n:
        urea_n_diff:
        uric_acid:
        uric_acid_diff:
        admit_wt:
        admit_wt_diff:
        glucose_(70-105):
        glucose_(70-105)_diff:
        heart_rate:
        heart_rate_diff:
        hematocrit:
        hematocrit_diff:
        hemoglobin:
        hemoglobin_diff:
        magnesium_(1.6-2.6):
        magnesium_(1.6-2.6)_diff:
        phosphorous(2.7-4.5):
        phosphorous(2.7-4.5)_diff:
        potassium_(3.5-5.3):
        potassium_(3.5-5.3)_diff:
        respiratory_rate:
        respiratory_rate_diff:
        spo2:
        spo2_diff:
        temperature_c_(calc):
        temperature_c_(calc)_diff:
    """
    lab_events = get_processed_lab_events(use_cache=use_cache)
    chart_events = get_processed_chart_events(use_cache=use_cache)
    patients = get_processed_patient_info(use_cache=use_cache)
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

    outcome_df = get_death_outcome(current_merged_df, death_time_frame=3)
    current_merged_df = pd.merge(
        current_merged_df,
        outcome_df,
        how="inner"
    )

    return current_merged_df
