
from sklearn.ensemble import RandomForestClassifier


class Predictor(object):
    """Model used by the DecisionEngine to predict the probability of a patient living.

    Note:
        This class is a proxy to a sklearn model. The reason for this was to provide an abstraction so that
        changes to the model would not require code in the DecisionEngine
    """

    def __init__(self):
        self._model = \
            RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=12, max_features=None, n_estimators=40)

    def __getattr__(self, name):
        return getattr(self._model, name)
