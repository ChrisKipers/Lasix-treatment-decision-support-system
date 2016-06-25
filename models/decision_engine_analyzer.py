import pandas as pd
import logging

class DecisionEngineAnalyzer(object):

    def __init__(self, decision_engine, data):
        self._data = data
        self._decision_engine = decision_engine
        self._top_suggestions = self._decision_engine.get_treatment_suggestion(self._data)

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
        suggested_living = self._top_suggestions.probability_of_living >= .5
        suggested_living_df = pd.DataFrame({"treatment": self._top_suggestions.treatment, "alive": suggested_living})
        suggested_living_percent_by_treatment = suggested_living_df.pivot_table(values="alive", index="treatment")

        actual_died_counts_by_treatment = self._data.pivot_table(values="died", index="treatment")
        actual_living_counts_by_treatment = 1 - actual_died_counts_by_treatment

        return pd.DataFrame({
            "suggested_survival_rate": suggested_living_percent_by_treatment,
            "actual_survival_rate": actual_living_counts_by_treatment
        })
