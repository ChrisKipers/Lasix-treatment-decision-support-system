import pandas as pd
import logging

from data_loading.data_loaders import get_lasix_poe, get_hospital_admissions
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.processed_data_interface import cache_results

@cache_results("lasix_poe.csv")
def get_processed_lasix(use_cache=False):
    """Processes the lasix poe data into a format that can be used for machine learning models.

    There are two major parts to the transformation:
        Columns dose_val_rx, dose_unit_rx and route are concatenated to create a treatment category.
        Treatments are extrapolated across the hospital admission dates. The extrapolation is done by first
        computing the date range for the hospital admission. Then for each date in the range we look to see
        what the treatment was for that day. Sometimes treatments overlap, because the previous treatment was
        cancelled prematurely in favor of a new treatment. In that scenario the newer treatment is used for
        that day. It is possible for days to have no treatment, in that case the value would be None.

    Note:
        Some hospital admissions do not have a lasix treatment. More investigation needs to be done to see why
        there are no treatments for them.

    Args:
        use_cache: Skip computation and load results from previous computation.

    Returns:
        A DataFrame with the following columns:
        date: The day of the treatment as a timestamp
        treatment: The treatment category for the day
        subject_id: The subject's ID
        hadm_id: The hospital admission ID
    """
    lasix_poe = get_lasix_poe()
    lasix_poe_w_dates = lasix_poe.dropna(subset=["start_dt", "stop_dt"])

    treatment_categories = \
        lasix_poe_w_dates.dose_val_rx + " " + lasix_poe_w_dates.dose_unit_rx + " " + lasix_poe_w_dates.route
    treatment_categories = treatment_categories.str.lower()
    treatment_categories.name = "treatment_category"

    # TODO: consider filtering out rows that have a rare treatment category.

    modify_dates_fn = get_modify_dates_fn()
    lasix_poe_w_dates = modify_dates_fn(lasix_poe_w_dates, ["start_dt", "stop_dt"])

    hospital_admissions = modify_dates_fn(get_hospital_admissions(), ["admit_dt", "disch_dt"])

    treatment_df = pd.concat([lasix_poe_w_dates, treatment_categories], axis=1)
    treatment_df_by_hadm_id = treatment_df.groupby("hadm_id")

    expanded_treatments = []
    hadm_ids_w_no_treatments = []
    for adm_row in hospital_admissions.itertuples():
        try:
            treatments_for_admission = treatment_df_by_hadm_id.get_group(adm_row.hadm_id).sort_values(["start_dt"])
        except:
            hadm_ids_w_no_treatments.append(adm_row.hadm_id)
            continue

        for day in pd.date_range(adm_row.admit_dt, adm_row.disch_dt):
            selected_row = None
            for current_row in treatments_for_admission.itertuples():
                if current_row.start_dt <= day <= current_row.stop_dt:
                    selected_row = current_row

            treatment_category = None if selected_row is None else selected_row.treatment_category

            expanded_treatments.append({
                "date": day,
                "treatment": treatment_category,
                "subject_id": adm_row.subject_id,
                "hadm_id": adm_row.hadm_id})

    logging.info("No treatments for %d hadm_ids: %s" % \
                 (len(hadm_ids_w_no_treatments), ",".join([str(s) for s in hadm_ids_w_no_treatments])))
    expanded_treatments_df = pd.DataFrame(expanded_treatments)
    expanded_treatments_df.treatment.fillna("No treatment", inplace=True)
    return expanded_treatments_df
