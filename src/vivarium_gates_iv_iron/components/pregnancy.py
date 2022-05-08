import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.components.hemoglobin import Hemoglobin
from vivarium_gates_iv_iron.components.mortality import MaternalMortality
from vivarium_gates_iv_iron.components.disability import MaternalDisability

from vivarium_gates_iv_iron.constants import models, data_keys
from vivarium_gates_iv_iron.constants.data_values import (
    DURATIONS,
)
from vivarium_gates_iv_iron.components.utilities import (
    load_and_unstack,
)


class Pregnancy:

    def __init__(self):
        self.new_children = NewChildren()

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    @property
    def sub_components(self):
        return [
            Hemoglobin(),
            MaternalMortality(),
            MaternalDisability(),
            self.new_children,
        ]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.columns_created = [
            'pregnancy_status',
            'pregnancy_outcome',
            'pregnancy_state_change_date',
            'pregnancy_duration',
            'maternal_hemorrhage',
        ] + self.new_children.columns_created

        self.prevalence = builder.lookup.build_table(
            load_and_unstack(builder, data_keys.PREGNANCY.PREVALENCE, 'pregnancy_status'),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        self.conception_rate = builder.value.register_rate_producer(
            'conception_rate',
            source=builder.lookup.build_table(
                builder.data.load(data_keys.PREGNANCY.CONCEPTION_RATE),
                key_columns=['sex'],
                parameter_columns=['age', 'year'],
            ),
        )
        self.outcome_probabilities = builder.lookup.build_table(
            load_and_unstack(
                builder,
                data_keys.PREGNANCY.CHILD_OUTCOME_PROBABILITIES,
                'pregnancy_outcome'
            ),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        self.probability_non_fatal_maternal_disorder = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.PROBABILITY_NONFATAL),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        self.probability_maternal_hemorrhage = builder.lookup.build_table(
            load_and_unstack(
                builder,
                data_keys.MATERNAL_DISORDERS.PROBABILITY_HEMORRHAGE,
                'hemorrhage_status'
            ),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        self.correction_factors = builder.lookup.build_table(
            load_and_unstack(
                builder,
                data_keys.PREGNANCY.HEMOGLOBIN_CORRECTION_FACTORS,
                'parameter'
            ),
            key_columns=['sex', 'pregnancy_status'],
            parameter_columns=['age', 'year'],
        )

        builder.value.register_value_modifier(
            "hemoglobin.exposure_parameters",
            self.hemoglobin_pregnancy_adjustment,
            requires_columns=["pregnancy_status"]
        )

        view_columns = self.columns_created + ['alive', 'exit_time', 'cause_of_death']
        self.population_view = builder.population.get_view(view_columns)
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_streams=[self.name],
            requires_columns=['age', 'sex'],
        )

        builder.event.register_listener("time_step", self.on_time_step)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        prevalence = self.prevalence(pop_data.index)
        pregnancy_status = self.randomness.choice(
            pop_data.index,
            choices=prevalence.columns.tolist(),
            p=prevalence,
            additional_key='pregnancy_status',
        )

        is_pregnant = pop_data.index[pregnancy_status != models.NOT_PREGNANT_STATE]

        child_status = self.new_children.empty(pop_data.index)
        child_status.loc[is_pregnant] = self.new_children(is_pregnant)

        maternal_status = pd.DataFrame({
            'pregnancy_status': pregnancy_status,
            'pregnancy_outcome': models.INVALID_OUTCOME,
            'pregnancy_duration': pd.NaT,
            'pregnancy_state_change_date': pd.NaT,
            'maternal_hemorrhage': models.NOT_MATERNAL_HEMORRHAGE_STATE,
        })

        p_outcome = self.outcome_probabilities(is_pregnant)
        maternal_status.loc[is_pregnant, 'pregnancy_outcome'] = self.randomness.choice(
            is_pregnant,
            choices=p_outcome.columns.tolist(),
            p=p_outcome,
            additional_key='pregnancy_outcome',
        )

        maternal_status.loc[is_pregnant, 'pregnancy_duration'] = pd.Timedelta(days=9 * 28)
        maternal_status.loc[:, 'pregnancy_state_change_date'] = (
            self._sample_initial_pregnancy_state_change_date(
                pregnancy_status,
                maternal_status['pregnancy_duration'],
                pop_data.creation_time
            )
        )

        pop_update = pd.concat([maternal_status, child_status], axis=1)
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event):
        pop_update = pd.concat([
            self._sample_new_pregnant(event.index, event.time),
            self._sample_new_prepostpartum(event.index, event.time),
            self._sample_new_postpartum(event.index, event.time),
            self._sample_new_not_pregnant(event.index, event.time),
        ])
        # Fix bug that causes pandas to coerce timedelta64[ns] to
        # datetime64[ns] when the whole column is set to NaT.
        pop_update['pregnancy_duration'] = pd.to_timedelta(pop_update['pregnancy_duration'])
        self.population_view.update(pop_update)

    def hemoglobin_pregnancy_adjustment(self, index: pd.Index, df: pd.DataFrame) -> pd.DataFrame:
        return df * self.correction_factors(index)

    ####################
    # Sampling helpers #
    ####################

    def _sample_initial_pregnancy_state_change_date(
        self,
        pregnancy_status: pd.Series,
        pregnancy_duration: pd.Series,
        creation_time: pd.Timestamp,
    ) -> pd.Series:
        index = pregnancy_status.index
        date = pd.Series(pd.NaT, index=pregnancy_status.index)

        is_pregnant = index[pregnancy_status == models.PREGNANT_STATE]

        draw = self.randomness.get_draw(index, additional_key='conception_date')
        days_until_pregnancy_ends = pregnancy_duration * draw
        conception_date = creation_time - days_until_pregnancy_ends
        date.loc[is_pregnant] = conception_date.loc[is_pregnant]

        is_postpartum = index[pregnancy_status == models.POSTPARTUM_STATE]

        draw = self.randomness.get_draw(index, additional_key='days_until_postpartum_ends')
        days_until_postpartum_ends = pd.to_timedelta(7 * DURATIONS.POSTPARTUM * draw)
        postpartum_start_date = creation_time - days_until_postpartum_ends
        date.loc[is_postpartum] = postpartum_start_date.loc[is_postpartum]

        is_prepostpartum = index[pregnancy_status == models.NO_MATERNAL_DISORDER_STATE]
        date.loc[is_prepostpartum] = (
            creation_time - pd.Timedelta(days=7 * DURATIONS.PREPOSTPARTUM)
        )
        return date

    def _sample_new_pregnant(self, index: pd.Index, event_time: pd.Timestamp) -> pd.DataFrame:
        pop = self.population_view.get(index)
        eligible = (pop.alive == 'alive') | (pop.exit_time == event_time)

        # Find the new mothers
        not_pregnant = pop[
            eligible & (pop['pregnancy_status'] == models.NOT_PREGNANT_STATE)
        ].index
        potentially_pregnant = self.randomness.filter_for_rate(
            pop.index,
            self.conception_rate(pop.index),
            additional_key='new_pregnancy'
        )
        newly_pregnant = pop.loc[not_pregnant.intersection(potentially_pregnant)].copy()

        newly_pregnant['pregnancy_status'] = models.PREGNANT_STATE

        p_outcome = self.outcome_probabilities(newly_pregnant.index)
        newly_pregnant['pregnancy_outcome'] = self.randomness.choice(
            newly_pregnant.index,
            choices=p_outcome.columns.tolist(),
            p=p_outcome,
            additional_key='pregnancy_outcome',
        )

        newly_pregnant['pregnancy_duration'] = pd.Timedelta(days=9 * 28)
        newly_pregnant['pregnancy_state_change_date'] = event_time
        newly_pregnant['maternal_hemorrhage'] = models.NOT_MATERNAL_HEMORRHAGE_STATE

        child_cols = self.new_children.columns_created
        newly_pregnant.loc[:, child_cols] = self.new_children(newly_pregnant.index)

        return newly_pregnant

    def _sample_new_prepostpartum(self, index: pd.Index, event_time: pd.Timestamp) -> pd.DataFrame:
        pop = self.population_view.get(index)
        eligible = (pop.alive == 'alive') | (pop.exit_time == event_time)

        # Find the newly prepostpartum
        new_prepostpartum = pop.loc[
            eligible
            & (pop['pregnancy_status'] == models.PREGNANT_STATE)
            & (event_time - pop["pregnancy_state_change_date"] >= pop["pregnancy_duration"])
        ].copy()

        # Check for maternal disorders and update
        fatal_md = new_prepostpartum[
            new_prepostpartum['cause_of_death'] == 'maternal_disorders'
        ].index
        non_fatal_md = self.randomness.filter_for_probability(
            new_prepostpartum.index,
            self.probability_non_fatal_maternal_disorder(new_prepostpartum.index),
            additional_key='maternal_disorder_incidence',
        )
        md = fatal_md.union(non_fatal_md)
        new_prepostpartum['pregnancy_status'] = models.NO_MATERNAL_DISORDER_STATE
        new_prepostpartum.loc[md, 'pregnancy_status'] = models.MATERNAL_DISORDER_STATE

        # Check for hemorrhage and update
        p_hemorrhage = self.probability_maternal_hemorrhage(new_prepostpartum.index)
        new_prepostpartum['maternal_hemorrhage'] = self.randomness.choice(
            new_prepostpartum.index,
            choices=p_hemorrhage.columns.tolist(),
            p=p_hemorrhage,
            additional_key='maternal_hemorrhage'
        )

        # Update event time
        new_prepostpartum['pregnancy_state_change_date'] = event_time
        return new_prepostpartum

    def _sample_new_postpartum(self, index: pd.Index, event_time: pd.Timestamp) -> pd.DataFrame:
        pop = self.population_view.get(index)
        eligible = (pop.alive == 'alive') | (pop.exit_time == event_time)

        prepostpartum_duration = pd.Timedelta(days=7 * DURATIONS.PREPOSTPARTUM)
        new_postpartum = pop.loc[
            eligible
            & pop['pregnancy_status'].isin(models.PREPOSTPARTUM_STATES)
            & (event_time - pop["pregnancy_state_change_date"] >= prepostpartum_duration)
        ].copy()

        new_postpartum["pregnancy_status"] = models.POSTPARTUM_STATE
        new_postpartum["pregnancy_state_change_date"] = event_time
        return new_postpartum

    def _sample_new_not_pregnant(self, index: pd.Index, event_time: pd.Timestamp) -> pd.DataFrame:
        pop = self.population_view.get(index)
        eligible = (pop.alive == 'alive') | (pop.exit_time == event_time)

        postpartum_duration = pd.Timedelta(days=7 * DURATIONS.POSTPARTUM)
        new_not_pregnant = pop.loc[
            eligible
            & (pop['pregnancy_status'] == models.POSTPARTUM_STATE)
            & (event_time - pop["pregnancy_state_change_date"] >= postpartum_duration)
        ].copy()
        # Postpartum to Not pregnant
        new_not_pregnant["pregnancy_status"] = models.NOT_PREGNANT_STATE
        new_not_pregnant["pregnancy_outcome"] = models.INVALID_OUTCOME
        new_not_pregnant["pregnancy_duration"] = pd.NaT
        new_not_pregnant["pregnancy_state_change_date"] = event_time
        new_not_pregnant["maternal_hemorrhage"] = models.NOT_MATERNAL_HEMORRHAGE_STATE

        new_not_pregnant.loc[:, self.new_children.columns_created] = (
            self.new_children.empty(new_not_pregnant.index)
        )
        return new_not_pregnant


class NewChildren:

    @property
    def name(self):
        return 'child_status'

    @property
    def columns_created(self):
        return ['sex_of_child', 'birth_weight']

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

    def empty(self, index: pd.Index):
        return pd.DataFrame({
            'sex_of_child': models.INVALID_OUTCOME,
            'birth_weight': np.nan,
        }, index=index)

    def __call__(self, index: pd.Index):
        sex_of_child = self.randomness.choice(
            index,
            choices=['Male', 'Female'],
            additional_key='sex_of_child',
        )
        # TODO implement LBWSG on next line for sampling
        draw = self.randomness.get_draw(index, additional_key='birth_weight')
        birth_weight = 1500. * (1 + draw)

        return pd.DataFrame({
            'sex_of_child': sex_of_child,
            'birth_weight': birth_weight,
        }, index=index)
