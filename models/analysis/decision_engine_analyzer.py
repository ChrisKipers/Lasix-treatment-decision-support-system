import os
import pandas as pd
import numpy as np
import shutil

ANALYSIS_RESULTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "__analysis_results__")


class DecisionEngineAnalyzer(object):
    def __init__(self, decision_engine, data):
        self._data = data
        self._decision_engine = decision_engine
        self._top_suggestions = self._decision_engine.get_treatment_suggestion(self._data)
        self._actual_treatment_with_recommended_treatment = \
            self.get_actual_treatment_with_recommended_treatment()

    def get_actual_treatment_with_recommended_treatment(self):
        actual = self._data[['treatment', 'died']]
        actual_survived_probability = self._decision_engine.get_probability_of_survival(self._data)
        actual_outcome = ~actual.died
        actual_outcome_prediction = actual_survived_probability >= 0.5

        recommended = self._top_suggestions[['treatment', 'probability_of_living']]
        recommended_outcome_prediction = recommended.probability_of_living >= 0.5

        return pd.DataFrame({
            "actual_treatment": actual.treatment.values,
            "actual_treatment_survived_probability": actual_survived_probability.values,
            "actual_survived": actual_outcome.values,
            "actual_treatment_survived_prediction": actual_outcome_prediction,
            "recommended_treatment": recommended.treatment.values,
            "recommended_treatment_survived_probability": recommended.probability_of_living.values,
            "recommended_treatment_survived_prediction": recommended_outcome_prediction.values
        })

    def get_recommended_treatment_overview(self):
        treatment_counts = self._get_treatment_counts()
        grouped_r_treatments = self._actual_treatment_with_recommended_treatment.groupby('recommended_treatment')
        actual_survival_rate = grouped_r_treatments.apply(lambda x: x.actual_survived.sum() / len(x))
        actual_survival_rate.name = "actual_surival_rate"
        actual_treatment_predicted_survival_rate = \
            grouped_r_treatments.apply(lambda x: x.actual_treatment_survived_prediction.sum() / len(x))
        actual_treatment_predicted_survival_rate.name = "actual_treatment_predicted_survival_rate"
        recommmended_treatment_predicted_surival_rate = \
            grouped_r_treatments.apply(lambda x: x.recommended_treatment_survived_prediction.sum() / len(x))
        recommmended_treatment_predicted_surival_rate.name = "recommended_treatment_predicted_survival_rate"
        survival_prediction_improvement = \
            recommmended_treatment_predicted_surival_rate - actual_treatment_predicted_survival_rate
        survival_prediction_improvement.name = "predicted_survival_rate_improvement"
        percent_actual_treatment_same_as_recommended = \
            grouped_r_treatments.apply(lambda x: (x.actual_treatment == x.recommended_treatment).sum() / len(x))
        percent_actual_treatment_same_as_recommended.name = "percent_of_treatment_same"

        return pd.concat([
            treatment_counts, percent_actual_treatment_same_as_recommended, actual_survival_rate,
            actual_treatment_predicted_survival_rate, recommmended_treatment_predicted_surival_rate,
            survival_prediction_improvement], axis=1)

    def get_outcome_change_by_recommended_and_actual_treatment(self):
        actual_predicted_survived = self._actual_treatment_with_recommended_treatment.actual_treatment_survived_prediction.astype(
            int)
        recommended_predicted_survived = self._actual_treatment_with_recommended_treatment.recommended_treatment_survived_prediction.astype(
            int)
        survived_change_diffs = recommended_predicted_survived - actual_predicted_survived
        survived_change_diffs.name = "survival_rate_improvement"
        treatments = \
            self._actual_treatment_with_recommended_treatment[
                ['recommended_treatment', 'actual_treatment', 'actual_survived',
                 'actual_treatment_survived_prediction']]
        treatments_with_change_diffs = pd.concat([treatments, survived_change_diffs], axis=1)
        counts = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values="survival_rate_improvement",
            aggfunc=len
        )
        counts.name = "counts"

        survival_rate_improvement = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values='survival_rate_improvement', aggfunc=np.mean
        )

        predicted_survival_rate = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values='actual_treatment_survived_prediction', aggfunc=np.mean
        )

        actual_survival_rate = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values='actual_survived', aggfunc=np.mean
        )

        merged_summary = pd.concat([counts, actual_survival_rate, predicted_survival_rate, survival_rate_improvement],
                                   axis=1).reset_index()

        value_counts_dict = self._data.treatment.value_counts().to_dict()
        merged_summary['percent_actual_treatment'] = merged_summary.counts / merged_summary.actual_treatment.map(
            value_counts_dict)
        column_order = ['recommended_treatment', 'actual_treatment', 'counts', 'percent_actual_treatment',
                        'actual_survived', 'actual_treatment_survived_prediction', 'survival_rate_improvement']
        return merged_summary[column_order]

    def get_dosage_difference(self):
        """IMPORTANT: The results do not account for differences in treatment route"""
        actual_treatment = self._actual_treatment_with_recommended_treatment.actual_treatment
        recommended_treatment = self._actual_treatment_with_recommended_treatment.recommended_treatment
        actual_dosage = pd.to_numeric(actual_treatment.str.extract('(\d+)')).fillna(0)
        recommended_dosage = pd.to_numeric(recommended_treatment.str.extract('(\d+)')).fillna(0)
        return recommended_dosage - actual_dosage

    def create_analysis_reports(self):
        if not os.path.exists(ANALYSIS_RESULTS_DIR):
            os.makedirs(ANALYSIS_RESULTS_DIR)

        self._decision_engine.get_outcome_feature_importance() \
            .sort_values('importance', ascending=False) \
            .to_csv(os.path.join(ANALYSIS_RESULTS_DIR, "outcome_feature_importance.csv"), index=False)

        print("Important Features for actual treatment prediction")
        self._decision_engine.get_actual_treatment_feature_importance() \
            .sort_values('importance', ascending=False) \
            .to_csv(os.path.join(ANALYSIS_RESULTS_DIR, "viable_treatment_feature_importance.csv"), index=False)

        self.get_recommended_treatment_overview() \
            .to_csv(os.path.join(ANALYSIS_RESULTS_DIR, "recommended_treatment_overview.csv"), index=False)

        actual_vs_recommended_treatment = self.get_outcome_change_by_recommended_and_actual_treatment()
        actual_vs_recommended_treatment \
            .to_csv(os.path.join(ANALYSIS_RESULTS_DIR, "recommended_vs_actual_treatment.csv"), index=False)

        top_treatment_improvements = \
            actual_vs_recommended_treatment[
                (actual_vs_recommended_treatment.counts > 20) &
                (actual_vs_recommended_treatment.survival_rate_improvement > 0.025)]
        top_treatment_improvements \
            .to_csv(os.path.join(ANALYSIS_RESULTS_DIR, "top_treatment_improvements.csv"), index=False)
        #
        # filtered_rto = recommended_treatment_overview[recommended_treatment_overview.suggested_count > 100]
        # filtered_rto.actual_count.plot.bar(title="Actual treatment distribution", rot=45)
        #
        # filtered_rto.suggested_count.plot.bar(title="Recommended treatment distribution", rot=45)
        #
        # filtered_rto.predicted_survival_rate_improvement.plot.bar(title="Predicted survival rate improvement", rot=45)

    def _get_treatment_counts(self):
        top_suggestion_treatment_counts = self._top_suggestions.treatment.value_counts()
        actual_treatment_counts = self._data.treatment.value_counts()
        return pd.DataFrame({
            "suggested_count": top_suggestion_treatment_counts,
            "actual_count": actual_treatment_counts
        })


def delete_previous_analysis_reports():
    if os.path.exists(ANALYSIS_RESULTS_DIR):
        shutil.rmtree(ANALYSIS_RESULTS_DIR)
