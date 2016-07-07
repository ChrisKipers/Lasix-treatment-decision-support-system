import os
import pickle
import logging

from sklearn.ensemble import RandomForestClassifier

from models.preprocess_pipeline import CongestiveHeartFailurePreprocessor
from models.decision_engine_predictors import OutcomePredictor, ActualTreatmentPredictor

from models.decision_engine import DecisionEngine

CACHE_MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "__cached_models__")
MODEL_FILE_PATH = os.path.join(CACHE_MODEL_DIR, "model.p")


def get_decision_engine(data):
    if os.path.exists(MODEL_FILE_PATH):
        logging.info("Loading decision engine from cache")
        return pickle.load(open(MODEL_FILE_PATH, 'rb'))
    else:
        if not os.path.exists(CACHE_MODEL_DIR):
            os.makedirs(CACHE_MODEL_DIR)
        logging.info("Creating decision engine")
        model = __get_decision_engine(data)
        pickle.dump(model, open(MODEL_FILE_PATH, 'wb'))
        return model


def delete_cached_model():
    if os.path.exists(MODEL_FILE_PATH):
        os.remove(MODEL_FILE_PATH)


def __get_decision_engine(data):
    survival_predictor = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=19, max_features=None,
                                                n_estimators=55)
    survival_preprocessor = CongestiveHeartFailurePreprocessor()
    outcome_predictor = OutcomePredictor(survival_predictor, survival_preprocessor)

    treatment_predictor = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=12, max_features=None,
                                                 n_estimators=40)
    treatment_preprocessor = CongestiveHeartFailurePreprocessor(False)
    actual_treatment_predictor = ActualTreatmentPredictor(treatment_predictor, treatment_preprocessor)

    return DecisionEngine(actual_treatment_predictor=actual_treatment_predictor,
                          outcome_predictor=outcome_predictor,
                          historical_data=data)
