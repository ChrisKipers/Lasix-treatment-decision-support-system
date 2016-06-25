import os
import pandas as pd

from sqlalchemy import create_engine

__CONGESTIVE_HEART_FAILURE_CODE = '428.0'

def get_lasix_poe():
    sql_query = """
    SELECT poe.*, poem.*
    FROM mimic2v26.poe_order as poe
    INNER JOIN mimic2v26.poe_med AS poem on poe.poe_id = poem.poe_id
    INNER JOIN mimic2v26.admissions AS a on a.hadm_id = poe.hadm_id
    INNER JOIN mimic2v26.icd9 as i on i.hadm_id = a.hadm_id
    WHERE i.code='%s' AND poe.medication = 'Furosemide'
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
    """ % __CONGESTIVE_HEART_FAILURE_CODE
    return get_query_results(sql_query)

def get_lab_item_details():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mimic2_reference_data", "lab_item_details.csv")
    return pd.read_csv(file_path)

# TODO: allow configuration
def get_query_results(sql_query):
    return pd.read_sql_query(sql_query, create_engine('postgresql://ckipers@localhost:5432/MIMIC2'), parse_dates=[])
