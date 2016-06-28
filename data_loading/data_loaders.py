import os
import pandas as pd

from sqlalchemy import create_engine

__CONGESTIVE_HEART_FAILURE_CODE = '428.0'

__TARGET_CHART_EVENTS = [
    7625, 1449, 51, 52, 3313, 3315, 3317, 3321, 3323, 3325, 5610, # Blood pressure
    211, 212, # Heart rate
    3580, 3581, 3582, 3583, 3693, 763,# Weight
    733, 3692, # Weight change
    1394, # Height
    618, 614, 615, 653, # Respiratory Rate
    646, # SpO2
    676, 677, 678, 679, # Temperature
    762, # Admit wt
    769, # Alt
    770, # AST
    772, # Albumin
    773, # Alk phosphate
    780, # Arterial pH ABG
    803, # Dire bili
    811, # Glucose
    813, # Hematocrit
    814, # Hemoglobin
    807, # Fingerstick glucose
    815, # INR
    817, # LDH
    821, # Mg
    824, # PT
    825, # PTT
    827, # Phosphorous
    829, # K
    834, # SaO2 ABG
    851, # Troponin
    853, # Uric acid
    861, # WBC
    7610, # Cardiac index
    7294, # BNP
    7246, # Cuff pressure
    7232, 7135, # Peakflow
    6256, # Platelet
    6255, 2445, 2334, # Vasopressin
    4354, # Bili,
    2697, 2699, # EF
    2681, 2338, # Fingerstick
    2394 # IVF
]

def get_lasix_poe():
    sql_query = """
    SELECT poe.*, poem.*
    FROM mimic2v26.poe_order as poe
    INNER JOIN mimic2v26.poe_med AS poem on poe.poe_id = poem.poe_id
    INNER JOIN mimic2v26.admissions AS a on a.hadm_id = poe.hadm_id
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = a.hadm_id
    WHERE i.code='%s' AND poe.medication = 'Furosemide' AND icustay_id IS NOT NULL
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
    # TODO: move list of target lab events to constants
    sql_query = """
    SELECT le.*
    FROM mimic2v26.labevents AS le
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = le.hadm_id
    WHERE i.code='%s' AND
    le.itemid IN (50159, 50090, 50177, 50195, 50073, 50062, 50188, 50189, 50384, 50386, 50277, 50178, 50149, 50383)
    AND icustay_id IS NOT NULL
    """ % __CONGESTIVE_HEART_FAILURE_CODE
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