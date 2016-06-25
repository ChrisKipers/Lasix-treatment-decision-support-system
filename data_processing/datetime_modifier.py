import pandas as pd
import functools

from data_loading.data_loaders import get_patients

__TARGET_YEAR = 2000

@functools.lru_cache()
def get_modify_dates_fn():
    """Returns a functions that can be used to transform date columns in a DataFrame by
    the subject's date offset."""
    offset_by_subject_id = __get_offset_by_subject_id()

    def fn(df, date_columns):
        """Transforms the date_columns in a data frame by the subject's date offset.

        Note:
            The data frame must contain a subject_id column and the date columns must be strings in the format
            %Y-%m-%d %H:%M:%S

        Args:
            df: Date frame with subject_id column.
            date_columns: Date columns that will be augmented by the subject's date offset then
            converted into datetime.

        Returns:
            The data frame augmented by modifying the date columns by the subjects date offset and then
            converting the columns to timestamp.
        """
        year_offsets_per_row = df.subject_id.map(offset_by_subject_id)
        for column in date_columns:
            modified_dates = []
            for (date, year_offset) in zip(df[column], year_offsets_per_row):
                new_year = date.year - year_offset
                modified_dates.append(date.replace(year=new_year))
            df[column] = pd.to_datetime(modified_dates)
        return df

    return fn


def __get_offset_by_subject_id():
    """Computes a year offset map that can be used to modify all dates related to a specific subject.
    Dates need to be modified because dates in MIMIC2 are obfuscated for anonymity, but the date ranges
    often fall outside of pandas max range. By modifying the dates we can utilize pandas datetime functionality.

    Returns:
        A dict mapping subject ID to the number of years in the offset. Example:
        {123478: 1249, 898281: -1000}
    """
    patients = get_patients()
    subject_ids = patients.subject_id
    year_as_int = pd.Series(patients.dob.apply(lambda x: x.year))
    # Must decrease in batches of 4 to preserve weird dates like leap year
    year_offset = ((year_as_int - __TARGET_YEAR) / 4).astype(int) * 4
    return {subject_id: offset for (subject_id, offset) in zip(subject_ids, year_offset)}
