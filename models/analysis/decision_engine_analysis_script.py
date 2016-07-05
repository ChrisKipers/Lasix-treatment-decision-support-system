import logging

from data_processing.ml_data_prepairer import get_ml_data
from models.build_decision_engine import get_random_forest_decision_engine
from models.analysis.decision_engine_analyzer import DecisionEngineAnalyzer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

congestive_heart_failure_data = get_ml_data()

decision_engine = get_random_forest_decision_engine(congestive_heart_failure_data)

decision_engine_analyzer = DecisionEngineAnalyzer(decision_engine, congestive_heart_failure_data)

print("Important Features for outcome prediction")
print(decision_engine.get_outcome_feature_importance().sort_values('importance', ascending=False))

print("Important Features for actual treatment prediction")
print(decision_engine.get_actual_treatment_feature_importance().sort_values('importance', ascending=False))

recommended_treatment_overview = decision_engine_analyzer.get_recommended_treatment_overview()
print('Recommended treatment overview')
print(recommended_treatment_overview)

outcome_changes = decision_engine_analyzer.get_outcome_change_by_recommended_and_actual_treatment()
print('Recommended treatment counts per actual treatment')
print(outcome_changes)

top_treatment_improvements = outcome_changes[(outcome_changes.counts > 20) & (outcome_changes.survival_rate_improvement > 0.025)]
print("Top opportunities for treatment improvements")
print(top_treatment_improvements.sort('counts', ascending=False))

filtered_rto = recommended_treatment_overview[recommended_treatment_overview.suggested_count > 100]
filtered_rto.actual_count.plot.bar(title="Actual treatment distribution", rot=45)

filtered_rto.suggested_count.plot.bar(title="Recommended treatment distribution", rot=45)

filtered_rto.predicted_survival_rate_improvement.plot.bar(title="Predicted survival rate improvement", rot=45)