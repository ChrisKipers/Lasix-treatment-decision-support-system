import pandas as pd


class OutcomePredictionAnalyzer(object):
    def __init__(self, outcome_prediction_results_df):
        self._outcome_prediction_results_df = outcome_prediction_results_df
        actual = self._outcome_prediction_results_df.died
        predicted = self._outcome_prediction_results_df.prediction
        self._confusion_matrix_indices = {
            "ALL_RESULTS": [True for _ in range(len(self._outcome_prediction_results_df))],
            "TRUE_POSITIVE": actual & predicted,
            "TRUE_NEGATIVE": ~actual & ~predicted,
            "FALSE_POSITIVE": ~actual & predicted,
            "FALSE_NEGATIVE": actual & ~predicted
        }

    def get_treatment_distributions_with_error_type(self):
        all_value_counts = self._outcome_prediction_results_df.treatment.value_counts()

        def get_df_for_confusion_matrix_entry(key, indices):
            entries = self._outcome_prediction_results_df[indices]
            value_counts = entries.treatment.value_counts()
            value_counts.name = key + "_COUNTS"
            value_percents = value_counts / len(entries)
            value_percents.name = key + "_DISTRIBUTION"
            value_percents.fillna(0, inplace=True)
            error_percent = value_counts / all_value_counts
            error_percent.name = key + "_PERCENT"
            error_percent.fillna(0, inplace=True)
            return pd.concat([value_counts, value_percents, error_percent], axis=1)

        dfs = [get_df_for_confusion_matrix_entry(key, indices)
               for key, indices in self._confusion_matrix_indices.items()]

        results_wo_accuracy = pd.concat(dfs, axis=1)

        accuracy_series = results_wo_accuracy.TRUE_POSITIVE_PERCENT + results_wo_accuracy.TRUE_NEGATIVE_PERCENT
        accuracy_series.name = "ACCURACY"

        column_order = [
            "ALL_RESULTS_COUNTS",
            "ALL_RESULTS_DISTRIBUTION",
            "ACCURACY",
            "TRUE_POSITIVE_COUNTS",
            "TRUE_POSITIVE_DISTRIBUTION",
            "TRUE_POSITIVE_PERCENT",
            "TRUE_NEGATIVE_COUNTS",
            "TRUE_NEGATIVE_DISTRIBUTION",
            "TRUE_NEGATIVE_PERCENT",
            "FALSE_POSITIVE_COUNTS",
            "FALSE_POSITIVE_DISTRIBUTION",
            "FALSE_POSITIVE_PERCENT",
            "FALSE_NEGATIVE_COUNTS",
            "FALSE_NEGATIVE_PERCENT",
            "FALSE_NEGATIVE_DISTRIBUTION",
        ]

        return pd.concat([accuracy_series, results_wo_accuracy], axis=1) \
            .sort("ALL_RESULTS_COUNTS", ascending=False)[column_order]
