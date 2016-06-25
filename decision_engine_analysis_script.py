import logging
import pandas as pd

from data_processing.ml_data_prepairer import get_ml_data
from models.build_decision_engine import get_random_forest_decision_engine
from models.decision_engine_analyzer import DecisionEngineAnalyzer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

congestive_heart_failure_data = get_ml_data()

decision_engine = get_random_forest_decision_engine(congestive_heart_failure_data)

decision_engine_analyzer = DecisionEngineAnalyzer(decision_engine, congestive_heart_failure_data)

print("Important Features for outcome prediction")
print(decision_engine.get_outcome_feature_importance().sort_values('importance', ascending=False))

print("Important Features for actual treatment prediction")
print(decision_engine.get_actual_treatment_feature_importance().sort_values('importance', ascending=False))

print("%.3f%% of actual treatments match suggested treatments" %
      decision_engine_analyzer.get_percent_of_correct_treatments())

counts = decision_engine_analyzer.get_treatment_counts()
survival_rates = decision_engine_analyzer.get_treatment_survival_rate()
all_date = pd.merge(counts, survival_rates, left_index=True, right_index=True).sort_values('suggested_count', ascending=False)
print('Treatment suggestions')
print(all_date)

print('Actual treatment survival rate increase over recommended treatment')
print(decision_engine_analyzer.get_outcome_change_per_actual_treatment())

print('Recommended treatment survival rate increase over actual treatment')
print(decision_engine_analyzer.get_outcome_change_per_recommended_treatment())

print('Recommended treatment counts per actual treatment')
print(decision_engine_analyzer.get_recommended_treatments_count_per_actual_treatment())

print('Actual treatment counts per recommended treatment')
print(decision_engine_analyzer.get_actual_treatments_count_per_recommended_treatment())