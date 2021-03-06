import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Imputer
from sklearn.pipeline import Pipeline

from data_processing.chart_event_processor import ALL_CHART_ITEM_FIELDS
from data_processing.lab_event_processor import ALL_LAB_ITEM_FIELDS


_TREATMENT_FIELD = 'treatment'

_CATEGORY_FIELDS = ['sex']

_SCALAR_FIELDS = ALL_CHART_ITEM_FIELDS + ALL_LAB_ITEM_FIELDS + ["age"]

class CongestiveHeartFailurePreprocessor(object):
    """Prepares the congestive heart failure data to be used with a machine learning model, as well as
    provides functionality to relate feature importance back to the preprocessed field name.

    There are three important preprocessing steps:

    1) Perform one hot encoding on categorical values.
    2) Impute data. Categorical data is imputed by replacing missing values with the most popular value in the
    data set. The only categorical field that is imputed differently is treatment. If a value is missing in treatment
    it is replaced by the string "No treatment", signaling that no treatment was given.
    3) Transform all scalar values so that they are standardized with a mean of 0 and a standard deviation of 1.
    Standardization is not necessary for all machine learning models, however it does not hurt any learning algorithms.
    """

    def __init__(self, include_treatment_as_predictor=True):
        self._all_category_fields = \
            [_TREATMENT_FIELD] + _CATEGORY_FIELDS if include_treatment_as_predictor else _CATEGORY_FIELDS
        self._label_binarizer_by_field_name = \
            {name: LabelBinarizer() for name in self._all_category_fields}

        label_binarizers = \
            [(name, [_ImputCategoricalValues(), self._label_binarizer_by_field_name[name]])
             for name in self._all_category_fields]
        standard_scalers = [(field_name, [_Reshape(), Imputer(), StandardScaler()]) for field_name in _SCALAR_FIELDS]

        # IMPORTANT: The order of the features is very important for the transform_feature_importance method to function
        # correctly.
        self._pipeline = Pipeline([
            ('dfmapper', DataFrameMapper(label_binarizers + standard_scalers))
        ])

    def fit(self, X, y=None):
        return self._pipeline.fit(X, y)

    def transform(self, X):
        return self._pipeline.transform(X)

    def transform_feature_importance(self, raw_feature_importance):
        len_of_categories = [len(self._label_binarizer_by_field_name[name].classes_)
                             for name in self._all_category_fields]
        len_of_scalar_features = [1 for _ in _SCALAR_FIELDS]

        all_lengths = len_of_categories + len_of_scalar_features
        splice_start_pos = np.cumsum([0] + all_lengths[:-1])
        slice_ranges_for_feature = [(start_pos, start_pos + length)
                                    for (start_pos, length) in zip(splice_start_pos, all_lengths)]

        importance_per_feature = [sum(raw_feature_importance[start:end])
                                  for (start, end) in slice_ranges_for_feature]
        # IMPORTANT: The order of the feature names must match the order the features are in in the pipeline.
        feature_names_in_order = self._all_category_fields + _SCALAR_FIELDS
        return pd.DataFrame({"feature": feature_names_in_order, "importance": importance_per_feature})

class _ImputCategoricalValues(object):
    """Imputes categorical values by replacing missing values with the most popular value in the
    data set"""
    def __init__(self):
        self._most_popular_value = None

    def fit(self, X, y=None):
        x_as_series = pd.Series(X)
        self._most_popular_value = x_as_series.describe().top
        return self

    def transform(self, X):
        x_as_series = pd.Series(X)
        return x_as_series.fillna(self._most_popular_value).values

class _Reshape(object):
    """Reshape an array so that it has an extra dimension of 1. This is required so that sklearn's Imputer
    works with the output of DataFrameMapper"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.shape = (X.shape[0], 1)
        return X
