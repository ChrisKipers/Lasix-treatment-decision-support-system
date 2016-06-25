from operator import itemgetter

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

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

    def __init__(self, survival_predictor, treatment_predictor, survival_preprocessor, treatment_preprocessor, historical_data):
        """

        Args:
            DataFrame with past patient data. The data is used to train the machine learning model.
        """
        self._treatment_categories = historical_data.treatment.unique()
        self._treatment_label_binarizer = LabelBinarizer()
        self._treatment_label_binarizer.fit(self._treatment_categories)
        self._survival_preprocessor = survival_preprocessor
        self._survival_predictor = survival_predictor
        self._treatment_preprocessor = treatment_preprocessor
        self._treatment_predictor = treatment_predictor
        self._survival_pipeline =\
            Pipeline([('preprocess', self._survival_preprocessor), ('predictor', self._survival_predictor)])
        self._treatment_predictor =\
            Pipeline([('preprocess', self._treatment_preprocessor), ('predictor', self._treatment_predictor)])

        self._train_survival_pipeline(historical_data)
        self._train_treatment_pipeline(historical_data)

    def get_treatment_suggestion(self, prediction_df):
        prediction_df = prediction_df.drop('treatment', axis=1).copy()
        prediction_df['sample_id'] = range(len(prediction_df))
        treatments_per_sample_id = self._get_treatments_per_sample_id(prediction_df)
        expanded_prediction_df = prediction_df.merge(treatments_per_sample_id, on="sample_id")

        # all_prediction_dfs = \
        #     [self._create_data_copy_with_treatment(prediction_df, treatment)
        #      for treatment in self._treatment_categories]
        # expanded_prediction_df = pd.concat(all_prediction_dfs)
        # expanded_prediction_df.reset_index(level=0, inplace=True, drop=True)

        # The predict probability returns an array of tuples. The first element in the tuple is probability of living,
        # the second element is probability of dying.
        expanded_prediction_df['probability_of_living'] = \
            [prob[0] for prob in self._survival_pipeline.predict_proba(expanded_prediction_df)]

        return self._get_rows_with_best_probility_for_sample_id(expanded_prediction_df)

    def _create_data_copy_with_treatment(self, data, treatment):
        data_copy = data.copy()
        data_copy['treatment'] = treatment
        return data_copy

    def _get_treatments_per_sample_id(self, prediction_df):
        # leave comment on structure
        probabilities_sectioned_by_treatment = self._treatment_predictor.predict_proba(prediction_df)
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
        # Perhaps consider choosing the top n treatment options
        #return combined_df[combined_df.probability_of_treatment > 0.03]

    def get_feature_importance(self):
        raw_feature_importance = self._survival_predictor.feature_importances_
        return self._survival_preprocessor.transform_feature_importance(raw_feature_importance)

    def _train_survival_pipeline(self, historical_data):
        y = historical_data.died.values
        X_train, X_test, y_train, y_test = train_test_split(historical_data, y, test_size=0.3)

        self._survival_pipeline.fit(X_train, y_train)

        train_pred = self._survival_pipeline.predict(X_train)
        print("survival train accuracy %.5f" % accuracy_score(train_pred, y_train))

        test_pred = self._survival_pipeline.predict(X_test)
        print("survival test accuracy %.5f" % accuracy_score(test_pred, y_test))

    def _train_treatment_pipeline(self, historical_data):
        y = self._treatment_label_binarizer.transform(historical_data.treatment.values)
        X_train, X_test, y_train, y_test = train_test_split(historical_data, y, test_size=0.3)

        self._treatment_predictor.fit(X_train, y_train)

        # TODO: use better accuracy measure, such as multiclass log loss
        train_pred = self._treatment_predictor.predict(X_train)
        print("Treatment train accuracy %.5f" % accuracy_score(train_pred, y_train))

        test_pred = self._treatment_predictor.predict(X_test)
        print("Treatment test accuracy %.5f" % accuracy_score(test_pred, y_test))

    def _get_rows_with_best_probility_for_sample_id(self, df):
        max_idx = df.groupby('sample_id').apply(lambda x: x['probability_of_living'].idxmax())
        df.reset_index(level=0, inplace=True, drop=True)
        return df.ix[max_idx]
