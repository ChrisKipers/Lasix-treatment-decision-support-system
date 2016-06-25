from data_loading.data_loaders import get_patients, get_demographic_details, get_hospital_admissions
from data_processing.datetime_modifier import get_modify_dates_fn
from data_processing.processed_data_interface import cache_results

@cache_results("patient_info.csv")
def get_patient_info(use_cache=False):
    """Returns non medical information about a patient for a hospital admission stay.

    Args:
        use_cache: Skip computation and load results from previous computation.

    Returns:
        A DataFrame with the following columns:
        subject_id:
        hadm_id:
        sex:
        marital_status_descr:
        ethnicity_descr:
        overall_payor_group_descr:
        religion_descr:
        age:
    TODO: Add comments for each column
    """
    modify_dates_fn = get_modify_dates_fn()
    patients = get_patients()
    admissions = get_hospital_admissions()
    demographic_details = get_demographic_details()

    target_patient_data = patients[["subject_id", "hadm_id", "sex", "dob"]]

    target_demographic_fields = [
        "subject_id",
        "hadm_id",
        "marital_status_descr",
        "ethnicity_descr",
        "overall_payor_group_descr",
        "religion_descr"
    ]

    target_demographic_details = demographic_details[target_demographic_fields]

    patients_info = target_patient_data.merge(target_demographic_details, on=["subject_id", "hadm_id"])

    admissions = modify_dates_fn(admissions, ["admit_dt"])
    patients_info = modify_dates_fn(patients_info, ["dob"])

    ages = (admissions.admit_dt - patients_info.dob).astype("timedelta64[Y]")
    patients_info.drop("dob", axis=1, inplace=True)
    patients_info["age"] = ages
    return patients_info
