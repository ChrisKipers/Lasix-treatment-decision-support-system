import pandas as pd
import logging

from data_loading.data_loaders import get_lasix_poe, get_icustay_details
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.processed_data_interface import cache_results


@cache_results("lasix_poe.csv")
def get_processed_lasix(use_cache=False):
    """Processes the lasix poe data into a format that can be used for machine learning models.

    There are two major parts to the transformation:
        Columns dose_val_rx, dose_unit_rx and route are concatenated to create a treatment category.
        Treatments are extrapolated across the icustay dates. The extrapolation is done by first
        computing the date range for the icustay. Then for each date in the range we look to see
        what the treatment was for that day. Sometimes treatments overlap, because the previous treatment was
        cancelled prematurely in favor of a new treatment. In that scenario the newer treatment is used for
        that day. It is possible for days to have no treatment, in that case the value would be None.

    Note:
        Some icustays do not have a lasix treatment. More investigation needs to be done to see why
        there are no treatments for them.

    Args:
        use_cache: Skip computation and load results from previous computation.

    Returns:
        A DataFrame with the following columns:
        date: The day of the treatment as a timestamp
        treatment: The treatment category for the day
        icustay_id: The icustay ID
    """
    lasix_poe = get_lasix_poe()
    lasix_poe_w_dates = lasix_poe.dropna(subset=["start_dt", "stop_dt"])

    treatment_categories = \
        lasix_poe_w_dates.dose_val_rx + " " + lasix_poe_w_dates.dose_unit_rx + " " + lasix_poe_w_dates.route
    # Standardize units since it doesn't make sence why "ml iv" is different from "mg iv"
    treatment_categories = treatment_categories.str.lower().str.replace("ml", "mg")
    treatment_categories.name = "treatment_category"

    # TODO: consider filtering out rows that have a rare treatment category.

    modify_dates_fn = get_modify_dates_fn()
    lasix_poe_w_dates = modify_dates_fn(lasix_poe_w_dates, ["start_dt", "stop_dt"])

    icu_details = modify_dates_fn(get_icustay_details(), ['icustay_intime', 'icustay_outtime'])

    treatment_df = pd.concat([lasix_poe_w_dates, treatment_categories], axis=1)
    treatment_df_by_icu_id = treatment_df.groupby("icustay_id")

    expanded_treatments = []
    icu_id_w_no_treatments = []
    for icu_row in icu_details.itertuples():
        try:
            treatments_for_icustay = treatment_df_by_icu_id.get_group(icu_row.icustay_id).sort_values(["start_dt"])
        except:
            icu_id_w_no_treatments.append(icu_row.icustay_id)
            icu_stay_time_delta = icu_row.icustay_outtime - icu_row.icustay_intime
            icu_stay_in_hours = (icu_stay_time_delta.days * 24) + icu_stay_time_delta.seconds * (60 * 60)

            # Only count the ICU if they meet an hour threshold. The threshold is required so that they are given the
            # oportunity to receive treatment. If we include records where they are denied treatment, then the
            # recommended treatment will be no treatment despite that not being the case since they died.

            if icu_stay_in_hours >= 12:
                # Create empty df so that each day in icustay will have no treatment
                treatments_for_icustay = pd.DataFrame()
            else:
                continue

        for day in pd.date_range(icu_row.icustay_intime.date(), icu_row.icustay_outtime.date()):
            selected_row = None
            for current_row in treatments_for_icustay.itertuples():
                if current_row.start_dt <= day <= current_row.stop_dt:
                    selected_row = current_row

            treatment_category = None if selected_row is None else selected_row.treatment_category
            expanded_treatments.append({
                "date": day,
                "treatment": treatment_category,
                "icustay_id": icu_row.icustay_id})

    logging.info("No treatments for %d icustay_id: %s" % \
                 (len(icu_id_w_no_treatments), ",".join([str(s) for s in icu_id_w_no_treatments])))
    expanded_treatments_df = pd.DataFrame(expanded_treatments)
    expanded_treatments_df.treatment.fillna("No treatment", inplace=True)
    return expanded_treatments_df
