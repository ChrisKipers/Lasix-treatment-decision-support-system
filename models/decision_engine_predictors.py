import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer


class _BasePredictor(object):
    """Provides shared functionally used by OutcomePredictor and ActualTreatmentPredictor"""

    def __init__(self, prediction_model, preprocessor):
        """

        Args:
            prediction_model: The model used to make predictions
            preprocessor: The preprocessor used to transform the patient features into a format that can be
            used by the predition_model

        """
        self._is_trained = False

        self._prediction_model = prediction_model
        self._preprocessor = preprocessor

        self._pipeline = \
            Pipeline([('preprocess', preprocessor), ('predictor', prediction_model)])

    def fit(self, data):
        """Trains the prediction model.

        Args:
            data: A dataframe containing patient features used to train the prediction model.

        """
        self._pre_fit_hook(data)

        y = self._get_outcome_data_for_training(data)
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

        self._pipeline.fit(X_train, y_train)

        train_pred = self._pipeline.predict(X_train)
        print("train accuracy %.5f" % accuracy_score(train_pred, y_train))

        test_pred = self._pipeline.predict(X_test)
        print("test accuracy %.5f" % accuracy_score(test_pred, y_test))

        self._is_trained = True

        return self

    def get_feature_importance(self):
        """Returns a dataframe containing the each column used by the prediction model and the relative importance
        it has to the outcome.

        Returns:
            A dataframe with the following columns:
            feature: The column
            importance: How important the column is in determining the prediction.
            The sum of all importances adds up to 1.
        """
        self._checked_is_trained()
        raw_feature_importance = self._prediction_model.feature_importances_
        return self._preprocessor.transform_feature_importance(raw_feature_importance)

    def _checked_is_trained(self):
        if not self._is_trained:
            raise ValueError("Not trained")

    def _get_outcome_data_for_training(self, data):
        pass

    def _pre_fit_hook(self, data):
        pass

class OutcomePredictor(_BasePredictor):
    """Predicts the probability the patient survived."""

    def get_probability_of_survival(self, data):
        """Returns the probability that the patient survived for each row in the dataframe.

        Args:
            data: Dataframe containing patient features.

        Returns:
            A series where the nth entry is the probability of survival for the nth row in the dataframe.

        """
        self._checked_is_trained()
        return pd.Series([prob[0] for prob in self._pipeline.predict_proba(data)])

    def _get_outcome_data_for_training(self, data):
        return data.died.values


class ActualTreatmentPredictor(_BasePredictor):
    """Returns the most likely treatments for a patient."""

    def __init__(self, prediction_model, preprocessor):
        super().__init__(prediction_model, preprocessor)

        self._treatment_label_binarizer = LabelBinarizer()

    def _pre_fit_hook(self, data):
        self._treatment_label_binarizer.fit(data.treatment.unique())

    def _get_outcome_data_for_training(self, data):
        return self._treatment_label_binarizer.transform(data.treatment.values)

    def get_possible_treatments(self, data):
        """Returns the most likely treatments for a patient.

        Args:
            data: A dataframe containing patient features as well as a sample_id column. The sample_id column
            is needed because there can be many most likely treatments for a sample_id and the column is used
            to reconcile the treatment with the record.

        Returns:
            A dataframe with the following columns:
            sample_id: The sample_id the treatment is for.
            treatment: The treatment category

        """
        self._checked_is_trained()

        # leave comment on structure
        probabilities_sectioned_by_treatment = self._pipeline.predict_proba(data)
        ordered_treatments = self._treatment_label_binarizer.classes_
        treatment_dfs = []

        for (treatment, probabilities_for_treatment) in zip(ordered_treatments, probabilities_sectioned_by_treatment):
            probability_of_treatment = [prob[1] if len(prob) > 1 else 0 for prob in probabilities_for_treatment]
            df = pd.DataFrame({
                "treatment": treatment,
                "probability_of_treatment": probability_of_treatment,
                "sample_id": range(len(probability_of_treatment))
            })
            treatment_dfs.append(df)

        combined_df = pd.concat(treatment_dfs)
        selected_treatement = combined_df.groupby("sample_id")["probability_of_treatment"].nlargest(5).reset_index().drop('level_1', axis=1)
        return pd.merge(combined_df, selected_treatement, left_on=["sample_id", "probability_of_treatment"], right_on=["sample_id", 0])[["sample_id", "treatment"]]
        #return combined_df[combined_df.probability_of_treatment > 0.03]