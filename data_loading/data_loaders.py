import os
import pandas as pd

from sqlalchemy import create_engine

__CONGESTIVE_HEART_FAILURE_CODE = '428.0'

__TARGET_LAB_ITEM_IDS = [
    50159, # Sodium in Blood
    50090, # Creatine
    50177, # Urea nitrogen
    50195, # BNP
    50073, # AST
    50062, # ALT
    50188, # Troponin I
    50189, # Troponin T
    50384, # Hematocrit
    50386, # Hemoglobin
    50277, # Sodium in urine
    50178, # Uric acid in blood
    50149, # Potassium
    50383, # Hematocrit
]

__TARGET_CHART_EVENTS = [
    211, # heart rate
    813, # hematrocrit
    814, # hemoglobin
    821, # magnesium
    827, # phosphorous
    618, # respiratory rate
    646, # spo2
    677, # temperature
    811, # glucose
    762 # admit_weight
]

def get_lasix_poe():
    sql_query = """
    SELECT poe.*, poem.*
    FROM mimic2v26.poe_order as poe
    INNER JOIN mimic2v26.poe_med AS poem on poe.poe_id = poem.poe_id
    INNER JOIN mimic2v26.admissions AS a on a.hadm_id = poe.hadm_id
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = a.hadm_id
    WHERE i.code='%s' AND poe.medication = 'Furosemide' AND poe.icustay_id IS NOT NULL
    """ % __CONGESTIVE_HEART_FAILURE_CODE
    return get_query_results(sql_query)


def get_hospital_admissions():
    sql_query = """
    SELECT a.* FROM mimic2v26.admissions AS a
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = a.hadm_id
    WHERE i.code='%s'
    """ % __CONGESTIVE_HEART_FAILURE_CODE
    return get_query_results(sql_query)


def get_patients():
    sql_query = """
    SELECT p.*, i.hadm_id
    FROM mimic2v26.icd9 as i
    INNER JOIN mimic2v26.d_patients AS p ON p.subject_id = i.subject_id
    WHERE i.code='%s'
    """ % __CONGESTIVE_HEART_FAILURE_CODE
    return get_query_results(sql_query)


def get_demographic_details():
    sql_query = """
    SELECT dd.*
    FROM mimic2v26.demographic_detail as dd
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = dd.hadm_id
    WHERE i.code='%s'
    """ % __CONGESTIVE_HEART_FAILURE_CODE
    return get_query_results(sql_query)


def get_lab_events():
    sql_query = """
    SELECT le.*
    FROM mimic2v26.labevents AS le
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = le.hadm_id
    WHERE i.code='%s' AND
    le.itemid IN (%s)
    AND icustay_id IS NOT NULL
    """ % (__CONGESTIVE_HEART_FAILURE_CODE, ", ".join([str(item) for item in __TARGET_LAB_ITEM_IDS]))
    lab_item_details = get_lab_item_details()[['itemid', 'test_name']].rename(columns={"test_name": "label"})
    lab_events = get_query_results(sql_query)
    return lab_events.merge(lab_item_details)

def get_chart_events():
    sql_query = """
    SELECT ce.*
    FROM mimic2v26.chartevents as ce
    INNER JOIN mimic2v26.icd9 as i on i.subject_id = ce.subject_id
    WHERE i.code='%s' AND ce.itemid IN (%s) AND icustay_id IS NOT NULL
    """ % (__CONGESTIVE_HEART_FAILURE_CODE, ", ".join([str(item) for item in __TARGET_CHART_EVENTS]))
    chart_items = get_query_results(sql_query)
    chart_item_details = _get_chart_item_details()[['itemid', 'label']]
    return chart_items.merge(chart_item_details)

def get_icustay_details():
    sql_query = """
    SELECT icd.*
    FROM mimic2v26.icustay_detail as icd
    INNER JOIN mimic2v26.icd9 as i on i.subject_id = icd.subject_id
    WHERE i.code='%s'
    """ % __CONGESTIVE_HEART_FAILURE_CODE
    return get_query_results(sql_query)

def get_lab_item_details():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mimic2_reference_data", "lab_item_details.csv")
    return pd.read_csv(file_path)

def _get_chart_item_details():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mimic2_reference_data", "chart_item_details.csv")
    return pd.read_csv(file_path)

# TODO: allow configuration
def get_query_results(sql_query):
    return pd.read_sql_query(sql_query, create_engine('postgresql://ckipers@localhost:5432/MIMIC2'), parse_dates=[])

# TODO remove or move to another file
def analyze_chart_items(chart_items_df):
    grouped_chart_items = chart_items_df.groupby('label')
    counts = grouped_chart_items.apply(lambda x: len(x))
    unique_subject_ids = grouped_chart_items.apply(lambda x: len(x.subject_id.unique()))
    units_of_measurement = grouped_chart_items.apply(lambda x: len(x.value1uom.unique()))
    results = pd.concat([counts, unique_subject_ids, units_of_measurement], axis=1)
    results.columns = ['num_of_results', 'unique_subject_ids', 'unique_units']
    return results

def get_poe_report(icustay_ids):
    sql_query = """
    select medication, count(medication) from mimic2v26.poe_order where icustay_id IN (%s) group by medication
    """ % ", ".join([str(item) for item in icustay_ids])
    return get_query_results(sql_query)