from typing import Dict

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.constants import (
    data_keys,
    models,
)


class MaternalInterventions:
    configuration_defaults = {
        'intervention': {
            'start_year': 2025,
            'scenario': 'baseline',
        }
    }

    @property
    def name(self) -> str:
        return 'maternal_interventions'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)
        self.coverage = self._load_intervention_coverage(builder)

        self.columns_required = [
            'pregnancy_status',
            'pregnancy_state_change_date',
            'maternal_bmi_anemia_category'
        ]
        self.columns_created = [
            'treatment_propensity',
            'maternal_supplementation',
            'antenatal_iv_iron',
            'postpartum_iv_iron',
        ]
        self.population_view = builder.population.get_view(
            self.columns_required + self.columns_created
        )
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_columns=self.columns_required,
            requires_streams=[self.name]
        )
        builder.event.register_listener(
            'time_step',
            self.on_time_step,
            priority=8,  # After pregnancy state changes
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pregnant, postpartum, in_treatment_window = self._get_indicators(
            pop_data.index, pop_data.creation_time,
        )
        propensity = self.randomness.get_draw(pop_data.index).rename('treatment_propensity')
        sampling_map = {
            'maternal_supplementation': (
                (pregnant & in_treatment_window(7 * 8)) | postpartum,
                ['ifa', 'other'],
                self._sample_oral_iron_status
             ),
            'antenatal_iv_iron': (
                (pregnant & in_treatment_window(7 * 15)) | postpartum,
                'antenatal_iv_iron',
                self._sample_iv_iron_status
            ),
            'postpartum_iv_iron': (
                postpartum,
                'postpartum_iv_iron',
                self._sample_iv_iron_status
            ),
        }
        pop_update = self._sample_intervention_status(
            propensity, pop_data.creation_time, sampling_map
        )
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        pregnant, postpartum, in_treatment_window = self._get_indicators(
            event.index, self.clock(), event.step_size,
        )
        sampling_map = {
            'maternal_supplementation': (
                pregnant & in_treatment_window(7 * 8),
                ['ifa', 'other'],
                self._sample_oral_iron_status
            ),
            'antenatal_iv_iron': (
                pregnant & in_treatment_window(7 * 15),
                'antenatal_iv_iron',
                self._sample_iv_iron_status
            ),
            'postpartum_iv_iron': (
                postpartum & in_treatment_window(7),
                'postpartum_iv_iron',
                self._sample_iv_iron_status),
        }
        pop_update = self._sample_intervention_status(
            pop['treatment_propensity'], event.time, sampling_map,
        )
        intervention_over = (
            (pop['pregnancy_status'] == models.NOT_PREGNANT_STATE)
            & (pop['pregnancy_state_change_date'] == event.time)
        )
        for intervention in ['maternal_supplementation',
                             'antenatal_iv_iron',
                             'postpartum_iv_iron']:
            pop_update.loc[intervention_over, intervention] = models.INVALID_TREATMENT

        self.population_view.update(pop_update)

    def _sample_intervention_status(
        self,
        propensity: pd.Series,
        time: pd.Timestamp,
        sampling_map: Dict
    ):
        index = propensity.index
        coverage = self._get_coverage(time)
        pop_update = [propensity]
        for name, (eligibility_mask, coverage_columns, sampling_func) in sampling_map.items():
            eligible = index[eligibility_mask]
            status = pd.Series(models.INVALID_TREATMENT, index=index, name=name)
            status.loc[eligible] = sampling_func(
                propensity.loc[eligible], coverage.loc[coverage_columns]
            )
            pop_update.append(status)
        pop_update = pd.concat(pop_update, axis=1)
        return pop_update

    def _sample_oral_iron_status(
        self,
        propensity: pd.Series,
        coverage: pd.Series,
    ) -> pd.Series:
        index = propensity.index
        bmi_status = self.population_view.subview(['maternal_bmi_anemia_category']).get(index)
        underweight = (bmi_status['maternal_bmi_anemia_category']
                       .isin([models.LOW_BMI_ANEMIC, models.LOW_BMI_NON_ANEMIC]))

        coverage.loc[models.NO_TREATMENT] = 1 - coverage.sum()
        p_covered = pd.DataFrame(coverage.to_dict(), index=index).cumsum(axis=1)
        choice_index = (propensity.values[np.newaxis].T > p_covered).sum(axis=1)

        supplementation = pd.Series(p_covered.columns[choice_index], index=index)
        other = supplementation == 'other'
        supplementation.loc[other & underweight] = models.BEP_SUPPLEMENTATION
        supplementation.loc[other & ~underweight] = models.MMS_SUPPLEMENTATION
        return supplementation

    def _sample_iv_iron_status(
        self,
        propensity: pd.Series,
        coverage: float,
    ) -> pd.Series:
        return pd.Series(
            np.where(propensity < coverage, models.TREATMENT, models.NO_TREATMENT),
            index=propensity.index
        )

    def _get_indicators(
        self,
        index: pd.Index,
        time: pd.Timestamp,
        step_size: pd.Timedelta = None,
    ):
        cols = ['pregnancy_status', 'pregnancy_state_change_date']
        pop = self.population_view.subview(cols).get(index)
        pregnant = pop['pregnancy_status'] == models.PREGNANT_STATE
        postpartum = pop['pregnancy_status'].isin([
            models.MATERNAL_DISORDER_STATE,
            models.NO_MATERNAL_DISORDER_STATE,
            models.POSTPARTUM_STATE,
        ])
        days_since_event = (time - pop['pregnancy_state_change_date']).dt.days

        if step_size is not None:
            # On time step. Check time to treat is within the current step.
            days_to_next_event = (
                time + step_size - pop['pregnancy_state_change_date']
            ).dt.days

            def in_window(time_to_treat: int):
                return ((days_since_event <= time_to_treat)
                        & (time_to_treat < days_to_next_event))
        else:
            # On initialize sims.  Check time to treat is in the past.
            def in_window(time_to_treat: int):
                return time_to_treat < days_since_event

        return pregnant, postpartum, in_window

    def _get_coverage(self, time: pd.Timestamp):
        if time < self.coverage.index.min():
            coverage = self.coverage.iloc[0]
        else:
            coverage = self.coverage.copy()
            coverage.loc[time, :] = np.nan
            coverage = coverage.sort_index().interpolate().loc[time]

        return coverage

    def _load_intervention_coverage(self, builder: Builder) -> pd.DataFrame:
        scenario = builder.configuration.intervention.scenario
        year_start = int(builder.configuration.intervention.start_year)

        data = builder.data.load(data_keys.MATERNAL_INTERVENTIONS.COVERAGE)
        data = data.set_index(['scenario', 'year', 'intervention']).unstack()
        data.columns = data.columns.droplevel()
        data = (data
                .reset_index()
                .drop(columns='mms')
                .rename(columns={'bep': 'other'}))

        data['time'] = (
            pd.Timestamp(f"{year_start}-1-1")
            + pd.to_timedelta(365.25 * (data['year'] - data['year'].min()), unit='D')
        )
        data = data.set_index(['scenario', 'time']).drop(columns='year')

        coverage = pd.concat([
            data.loc['baseline', 'ifa'].rename('baseline_ifa'),
            data.loc[scenario],
        ], axis=1)

        return coverage
