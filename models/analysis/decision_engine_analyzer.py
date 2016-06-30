import pandas as pd
import numpy as np
import logging

class DecisionEngineAnalyzer(object):

    def __init__(self, decision_engine, data):
        self._data = data
        self._decision_engine = decision_engine
        self._top_suggestions = self._decision_engine.get_treatment_suggestion(self._data)
        self._actual_treatment_with_recommended_treatment =\
            self.get_actual_treatment_with_recommended_treatment()

    def get_percent_of_correct_treatments(self):
        return (self._top_suggestions.treatment == self._data.treatment).sum() / len(self._data)

    def get_treatment_counts(self):
        top_suggestion_treatment_counts = self._top_suggestions.treatment.value_counts()
        actual_treatment_counts = self._data.treatment.value_counts()
        return pd.DataFrame({
            "suggested_count": top_suggestion_treatment_counts,
            "actual_count": actual_treatment_counts
        })

    def get_treatment_survival_rate(self):
        suggested_living = self._top_suggestions.probability_of_living >= 0.5
        suggested_living_df = pd.DataFrame({"treatment": self._top_suggestions.treatment, "alive": suggested_living})
        suggested_living_percent_by_treatment = suggested_living_df.pivot_table(values="alive", index="treatment")

        actual_died_counts_by_treatment = self._data.pivot_table(values="died", index="treatment")
        actual_living_counts_by_treatment = 1 - actual_died_counts_by_treatment

        return pd.DataFrame({
            "suggested_survival_rate": suggested_living_percent_by_treatment,
            "actual_survival_rate": actual_living_counts_by_treatment
        })

    def get_actual_treatment_with_recommended_treatment(self):
        actual = self._data[['treatment', 'died']]
        recommended = self._top_suggestions[['treatment', 'probability_of_living']]
        recommended_outcome = recommended.probability_of_living >= 0.5
        actual_outcome = ~actual.died

        return pd.DataFrame({
            "actual_treatment": actual.treatment.values,
            "actual_survived": actual_outcome.values,
            "recommended_treatment": recommended.treatment.values,
            "recommended_survived": recommended_outcome.values
        })

    def get_outcome_change_per_actual_treatment(self):
        a_survived = self._actual_treatment_with_recommended_treatment.actual_survived.astype(int)
        r_survived = self._actual_treatment_with_recommended_treatment.recommended_survived.astype(int)

        target_treatments = self._actual_treatment_with_recommended_treatment.actual_treatment.values
        survived_change_diffs = (a_survived - r_survived).values

        return self._get_outcome_change_per_treatment(target_treatments, survived_change_diffs)

    def get_outcome_change_per_recommended_treatment(self):
        a_survived = self._actual_treatment_with_recommended_treatment.actual_survived.astype(int)
        r_survived = self._actual_treatment_with_recommended_treatment.recommended_survived.astype(int)

        target_treatments = self._actual_treatment_with_recommended_treatment.recommended_treatment.values
        survived_change_diffs = (r_survived - a_survived).values

        return self._get_outcome_change_per_treatment(target_treatments, survived_change_diffs)

    def get_outcome_change_by_recommended_and_actual_treatment(self):
        a_survived = self._actual_treatment_with_recommended_treatment.actual_survived.astype(int)
        r_survived = self._actual_treatment_with_recommended_treatment.recommended_survived.astype(int)
        survived_change_diffs = r_survived - a_survived
        survived_change_diffs.name = "outcome_diff"
        treatments =\
            self._actual_treatment_with_recommended_treatment[['recommended_treatment', 'actual_treatment', 'actual_survived']]
        treatments_with_change_diffs = pd.concat([treatments, survived_change_diffs], axis=1)
        counts = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values="outcome_diff",
            aggfunc=len
        )
        counts.name = "counts"

        average_diff = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values='outcome_diff', aggfunc=np.mean
        )

        actual_survival_rate = pd.pivot_table(
            treatments_with_change_diffs,
            index=['recommended_treatment', 'actual_treatment'],
            values='actual_survived', aggfunc=np.mean
        )

        return pd.concat([counts, actual_survival_rate, average_diff], axis=1)

    def _get_outcome_change_per_treatment(self, treatments, survival_change_diff):
        change_with_actual_treatment = \
            pd.DataFrame({
                "treatment": treatments,
                "surv_change": survival_change_diff
            })

        return pd.pivot_table(change_with_actual_treatment, index='treatment', values='surv_change', aggfunc=np.mean)
