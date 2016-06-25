# Lasix Treatment Decision Engine

## Overview

The lasix treatment decision engine is designed to recommend the best lasix treatment for patient's suffering from
congestive heart failure based on the patient's current status. More information about the project can be found in the
[research paper](https://goo.gl/axDEmT).

## Project Structure

### data_loading

The data_loading package provides functionality for retrieving data from the MIMIC2 database.

### data_processing

The data_processing package provides functionality for transforming the MIMIC2 data into a tidy dataset that can be
used by the decision engine.

### models

The models package contains the code for the decision engine components.

### decision_engine_analysis_script.py

The decision_engine_analysis_script.py runs all the code to build the decision engine and evaluate it's recommendations
on the MIMIC2 data set.