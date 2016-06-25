
class DecisionEngine(object):
    """Provides lasix treatment recommendations based on patient data.

    The DecisionEngine works by training a machine learning model with training data learning how to predict
    if a patient lives or dies based on the patient's data and the treatment they were given.

    After the model is trained the model can then give its prediction based on patient's data including the treatment.
    The model can also give prediction probability instead of a hard prediction. This prediction probability is very
    important in the DecisionEngines recommendations.

    The DecisionEngine makes recommendations by taking a patients data and creating a combination of the data with
    every possible treatment. It then gets the prediction probabilities for every combination. It then merges the
    probabilities with the treatments and sorts them based on the highest probability of living. All the treatments
    along with the probabilities are returned the caller.
    """

    def __init__(self, actual_treatment_predictor, outcome_predictor, historical_data):
        """

        Args:
            DataFrame with past patient data. The data is used to train the machine learning model.
        """
        self._actual_treatment_predictor = actual_treatment_predictor
        self._outcome_predictor = outcome_predictor

        self._actual_treatment_predictor.fit(historical_data)
        self._outcome_predictor.fit(historical_data)

    def get_treatment_suggestion(self, prediction_df):
        # Remove treatment column from prediction_df since we will be cross referencing all valid treatments
        # with each patient feature.
        prediction_df = prediction_df.drop('treatment', axis=1).copy()
        # Add a sample_id column to the dataframe to be able to reconcile treatment recommendations with
        # the patient feature they are for.
        prediction_df['sample_id'] = range(len(prediction_df))

        # Get all valie treatments for each sample_id
        treatments_per_sample_id = self._actual_treatment_predictor.get_possible_treatments(prediction_df)
        # Cross reference all patient features with valid treatements for the features
        expanded_prediction_df = prediction_df.merge(treatments_per_sample_id, on="sample_id")

        # Add the probability of survival for each row
        expanded_prediction_df['probability_of_living'] = \
            self._outcome_predictor.get_probability_of_survival(expanded_prediction_df)

        # Keep only one record per sample_id. The record kept is the one with the highest probability.
        rows_with_best_probability = \
            self._get_rows_with_best_probility_for_sample_id(expanded_prediction_df)
        # Return only the treatment and probability_of_living. The returned dataframe can be matched with
        # the input dataframe by row position.
        return rows_with_best_probability[['treatment', 'probability_of_living']]

    def get_actual_treatment_feature_importance(self):
        """Returns a dataframe containing the each column used by the actual treatment prediction model
        and the relative importance it has to the outcome.

        Returns:
            A dataframe with the following columns:
            feature: The column
            importance: How important the column is in determining the prediction.
            The sum of all importances adds up to 1.
        """
        return self._actual_treatment_predictor.get_feature_importance()

    def get_outcome_feature_importance(self):
        """Returns a dataframe containing the each column used by the outcome prediction model
        and the relative importance it has to the outcome.

        Returns:
            A dataframe with the following columns:
            feature: The column
            importance: How important the column is in determining the prediction.
            The sum of all importances adds up to 1.
        """
        return self._outcome_predictor.get_feature_importance()

    def _get_rows_with_best_probility_for_sample_id(self, df):
        max_idx = df.groupby('sample_id').apply(lambda x: x['probability_of_living'].idxmax())
        df.reset_index(level=0, inplace=True, drop=True)
        return df.ix[max_idx]
