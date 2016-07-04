import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

from models.save_file_helper import save_debugging_file


class _BasePredictor(object):
    """Provides shared functionally used by OutcomePredictor and ActualTreatmentPredictor"""

    def __init__(self, prediction_model, preprocessor):
        """

        Args:
            prediction_model: The model used to make predictions
            preprocessor: The preprocessor used to transform the patient features into a format that can be
            used by the prediction_model

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
        class_name = self.__class__.__name__
        self._pre_fit_hook(data)

        y = self._get_outcome_data_for_training(data)
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1)

        self._pipeline.fit(X_train, y_train)

        train_pred = self._pipeline.predict(X_train)
        print("%s train accuracy %.5f" % (class_name, accuracy_score(train_pred, y_train)))

        test_pred = self._pipeline.predict(X_test)
        print("%s test accuracy %.5f" % (class_name, accuracy_score(test_pred, y_test)))

        has_treatment = data.treatment != 'No treatment'
        X_treatment = data[has_treatment]
        y_treatment = self._get_outcome_data_for_training(X_treatment)

        # Predict accuracy among the results with a treatment since the data is skewed so heavy in favor of no
        # treatment.
        treat_pred = self._pipeline.predict(X_treatment)
        print("%s treatment accuracy: %.5f" % (class_name, accuracy_score(treat_pred, y_treatment)))

        all_pred = self._pipeline.predict(data)
        predicted_value = self._get_predicted_value(all_pred)

        prediction_results = data.copy()
        prediction_results['prediction'] = predicted_value

        save_debugging_file(prediction_results, self.__class__.__name__ + "_prediction_results.csv")
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

    def _get_predicted_value(self, prediction):
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

    def _get_predicted_value(self, prediction):
        return prediction


class ActualTreatmentPredictor(_BasePredictor):
    """Returns the most likely treatments for a patient.

    Args:
        prediction_model: The model used to make predictions
        preprocessor: The preprocessor used to transform the patient features into a format that can be
        used by the prediction_model
        recommendation_probability_threshold: The probability threshold that a potential recommendation needs to have
        a higher probability than to be considered a possible treatment.
    """

    def __init__(self, prediction_model, preprocessor, recommendation_probability_threshold=0.05):
        super().__init__(prediction_model, preprocessor)

        self._treatment_label_binarizer = LabelBinarizer()
        self._recommendation_probability_threshold = recommendation_probability_threshold

    def _pre_fit_hook(self, data):
        self._treatment_label_binarizer.fit(data.treatment.unique())

    def _get_outcome_data_for_training(self, data):
        return self._treatment_label_binarizer.transform(data.treatment.values)

    def _get_predicted_value(self, prediction):
        return self._treatment_label_binarizer.inverse_transform(prediction)

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

        # Get all treatments that have a probability greater than the threshold
        sample_with_high_probability = \
            combined_df[combined_df.probability_of_treatment > self._recommendation_probability_threshold]

        # Get the top probability for a sample_id. This treatment will be used if there is no treatment for the
        # sample_id greater than the threshold
        top_treatment_per_sample_id = combined_df.groupby("sample_id")["probability_of_treatment"].nlargest(
            1).reset_index().drop('level_1', axis=1)

        # Find top treatments for samples that have not treatment above the threshold. This is a rare case but can
        # happen.
        samples_ids_with_high_prob = set(sample_with_high_probability.sample_id.unique())
        all_sample_ids = set(combined_df.sample_id.unique())
        ids_not_in_high_prob = all_sample_ids - samples_ids_with_high_prob

        top_treatments_for_samples_missing_high_prob =\
            top_treatment_per_sample_id[top_treatment_per_sample_id.sample_id.isin(ids_not_in_high_prob)]

        return pd.concat([sample_with_high_probability, top_treatments_for_samples_missing_high_prob])
