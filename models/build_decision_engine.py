from sklearn.ensemble import RandomForestClassifier

from models.preprocess_pipeline import CongestiveHeartFailurePreprocessor

from models.decision_engine import DecisionEngine

def get_random_forest_decision_engine(data):
    survival_predictor = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=12, max_features=None, n_estimators=40)
    survival_preprocessor = CongestiveHeartFailurePreprocessor()

    treatment_predictor = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=12, max_features=None, n_estimators=40)
    treatment_preprocessor = CongestiveHeartFailurePreprocessor(False)

    return DecisionEngine(survival_predictor=survival_predictor,
                          survival_preprocessor=survival_preprocessor,
                          treatment_predictor=treatment_predictor,
                          treatment_preprocessor=treatment_preprocessor,
                          historical_data=data)
