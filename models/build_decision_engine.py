from sklearn.ensemble import RandomForestClassifier

from models.preprocess_pipeline import CongestiveHeartFailurePreprocessor
from models.decision_engine_predictors import OutcomePredictor, ActualTreatmentPredictor

from models.decision_engine import DecisionEngine

def get_random_forest_decision_engine(data):
    survival_predictor = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=19, max_features=None, n_estimators=55)
    survival_preprocessor = CongestiveHeartFailurePreprocessor()
    outcome_predictor = OutcomePredictor(survival_predictor, survival_preprocessor)

    treatment_predictor = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=12, max_features=None, n_estimators=40)
    treatment_preprocessor = CongestiveHeartFailurePreprocessor(False)
    actual_treatment_predictor = ActualTreatmentPredictor(treatment_predictor, treatment_preprocessor)

    return DecisionEngine(actual_treatment_predictor=actual_treatment_predictor,
                          outcome_predictor=outcome_predictor,
                          historical_data=data)
