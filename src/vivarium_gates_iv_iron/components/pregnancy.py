import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.components.hemoglobin import Hemoglobin
from vivarium_gates_iv_iron.components.mortality import MaternalMortality

from vivarium_gates_iv_iron.constants import models, data_keys
from vivarium_gates_iv_iron.constants.data_values import (
    DURATIONS,
)
from vivarium_gates_iv_iron.components.utilities import (
    load_and_unstack,
)


class Pregnancy:
    def __init__(self):
        self.mortality = MaternalMortality()
        # self.hemoglobin_distribution = Hemoglobin()

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    @property
    def sub_components(self):
        # return [self.hemoglobin_distribution]
        return [
            self.mortality,
            # self.hemoglobin_distribution,
        ]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.columns_created = [
            'pregnancy_status',
            'pregnancy_outcome',
            'sex_of_child',
            'birth_weight',
            'pregnancy_state_change_date',
            'pregnancy_duration',
            'maternal_hemorrhage',
        ]

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

        self.ylds_per_maternal_disorder = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.YLDS),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        builder.value.register_value_modifier(
            "disability_weight",
            self.accrue_disability,
            requires_columns=["alive", "pregnancy_status"],
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

        view_columns = self.columns_created + ['alive', 'exit_time', 'age', 'sex']
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

        child_status = pd.DataFrame({
            'sex_of_child': models.INVALID_OUTCOME,
            'birth_weight': np.nan
        }, index=pop_data.index)
        child_status.loc[is_pregnant] = self._sample_child_outcomes(is_pregnant)

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
        pop = self.population_view.get(event.index, query="alive =='alive'")
        conception_rate = self.conception_rate(pop.index)

        pregnant_this_step = pd.Series(False, index=pop.index)
        pregnant_this_step_idx = self.randomness.filter_for_rate(pop.index, conception_rate, additional_key='new_pregnancy')
        pregnant_this_step.loc[pregnant_this_step_idx] = True
        pregnant_this_step = (pop['pregnancy_status'] == models.NOT_PREGNANT_STATE) & pregnant_this_step

        p = self.outcome_probabilities(pop.index)[list(models.PREGNANCY_OUTCOMES)]
        pregnancy_outcome = self.randomness.choice(pop.index, choices=models.PREGNANCY_OUTCOMES, p=p,
                                                   additional_key='pregnancy_outcome')

        sex_of_child = self.randomness.choice(pop.index, choices=['Male', 'Female'],
                                              p=[0.5, 0.5], additional_key='sex_of_child')

        # TODO: update with birth_weight distribution
        birth_weight = 1500.0 + 1500 * self.randomness.get_draw(pop.index, additional_key='birth_weight')

        pregnancy_duration = pd.to_timedelta(9 * 28,
                                             unit='d')

        # Make masks for subsets
        pregnancy_ends_this_step = (
                (pop['pregnancy_status'] == models.PREGNANT_STATE)
                & (event.time - pop["pregnancy_state_change_date"] >= pop["pregnancy_duration"])
        )
        maternal_disorder_incidence_draw = self.randomness.get_draw(pop.index,
                                                                    additional_key="maternal_disorder_incidence")
        maternal_disorder_this_step = maternal_disorder_incidence_draw < self.probability_non_fatal_maternal_disorder(
            pop.index)

        maternal_hemorrhage_incidence_draw = self.randomness.get_draw(pop.index,
                                                                      additional_key='maternal_hemorrhage_incidence')
        maternal_hemorrhage_this_step = maternal_hemorrhage_incidence_draw < self.probability_maternal_hemorrhage(
            pop.index)
        maternal_hemorrhage_severity_draw = self.randomness.get_draw(pop.index,
                                                                     additional_key="maternal_hemorrhage_severity_draw")
        moderate_maternal_hemorrhage_this_step = maternal_hemorrhage_severity_draw < self.maternal_hemorrhage_severity
        severe_maternal_hemorrhage_this_step = ~moderate_maternal_hemorrhage_this_step

        prepostpartum_ends_this_step = (

            (
                    ((pop['pregnancy_status'] == models.MATERNAL_DISORDER_STATE)
                     | (pop['pregnancy_status'] == models.NO_MATERNAL_DISORDER_STATE))
                    & (event.time - pop["pregnancy_state_change_date"] >=
                       pd.Timedelta(days=7 * DURATIONS.PREPOSTPARTUM))  # One time step
            )
        )
        postpartum_ends_this_step = (
                (pop['pregnancy_status'] == models.POSTPARTUM_STATE)
                & (event.time - pop["pregnancy_state_change_date"] >= pd.Timedelta(days=7 * DURATIONS.POSTPARTUM))
        )



        # Update new pregnancies
        # TODO: If you want to be mutually exclusive from death make this
        # pregnant_this_step = pregnant_this_step & ~died_this_step
        pop.loc[pregnant_this_step, "pregnancy_status"] = models.PREGNANT_STATE
        pop.loc[pregnant_this_step, "pregnancy_outcome"] = pregnancy_outcome.loc[pregnant_this_step]
        pop.loc[pregnant_this_step, "sex_of_child"] = sex_of_child.loc[pregnant_this_step]
        pop.loc[pregnant_this_step, "birth_weight"] = birth_weight.loc[pregnant_this_step]
        pop.loc[pregnant_this_step, "pregnancy_duration"] = pregnancy_duration
        pop.loc[pregnant_this_step, "pregnancy_state_change_date"] = event.time

        # Pregnancy to maternal disorder state and no maternal disorder state
        moderate_maternal_hemorrhage_this_step = (maternal_hemorrhage_this_step
                                                  & moderate_maternal_hemorrhage_this_step
                                                  & pregnancy_ends_this_step)
        severe_maternal_hemorrhage_this_step = (maternal_hemorrhage_this_step
                                                & severe_maternal_hemorrhage_this_step
                                                & pregnancy_ends_this_step)
        maternal_disorder_this_step = ((maternal_disorder_this_step
                                       | died_due_to_maternal_disorders)
                                       & pregnancy_ends_this_step)

        no_maternal_disorder_this_step = ~maternal_disorder_this_step & pregnancy_ends_this_step

        pop.loc[maternal_disorder_this_step, "pregnancy_status"] = models.MATERNAL_DISORDER_STATE
        pop.loc[maternal_disorder_this_step, "pregnancy_state_change_date"] = event.time

        pop.loc[severe_maternal_hemorrhage_this_step, 'maternal_hemorrhage'] = models.SEVERE_MATERNAL_HEMORRHAGE_STATE
        pop.loc[moderate_maternal_hemorrhage_this_step, 'maternal_hemorrhage'] = models.MODERATE_MATERNAL_HEMORRHAGE_STATE
        pop.loc[no_maternal_disorder_this_step, "pregnancy_status"] = models.NO_MATERNAL_DISORDER_STATE
        pop.loc[no_maternal_disorder_this_step, "pregnancy_state_change_date"] = event.time

        # Handle simulants going from (md or nmd) -> pp
        pop.loc[prepostpartum_ends_this_step, "pregnancy_status"] = models.POSTPARTUM_STATE
        pop.loc[prepostpartum_ends_this_step, "pregnancy_state_change_date"] = event.time

        # Postpartum to Not pregnant
        pop.loc[postpartum_ends_this_step, "pregnancy_status"] = models.NOT_PREGNANT_STATE
        pop.loc[postpartum_ends_this_step, "pregnancy_outcome"] = models.INVALID_OUTCOME
        pop.loc[postpartum_ends_this_step, "sex_of_child"] = models.INVALID_OUTCOME
        pop.loc[postpartum_ends_this_step, "birth_weight"] = np.nan
        pop.loc[postpartum_ends_this_step, "pregnancy_duration"] = pd.NaT
        pop.loc[postpartum_ends_this_step, "pregnancy_state_change_date"] = event.time
        pop.loc[postpartum_ends_this_step, "maternal_hemorrhage"] = models.NOT_MATERNAL_HEMORRHAGE_STATE

        self.population_view.update(pop)

    def accrue_disability(self, index: pd.Index):
        anemia_disability_weight = self.hemoglobin_distribution.disability_weight(index)
        maternal_disorder_ylds = self.ylds_per_maternal_disorder(index)
        maternal_disorder_disability_weight = maternal_disorder_ylds * 365 / self.step_size().days
        disability_weight = pd.Series(np.nan, index=index)

        pop = self.population_view.get(index)
        alive = pop["alive"] == "alive"
        pregnant_or_not_pregnant = alive & pop["pregnancy_status"].isin([models.PREGNANT_STATE, models.NOT_PREGNANT_STATE])
        maternal_disorders = alive & (pop["pregnancy_status"] == models.MATERNAL_DISORDER_STATE)
        no_maternal_disorders = alive & (pop["pregnancy_status"] == models.NO_MATERNAL_DISORDER_STATE)
        postpartum = alive & (pop["pregnancy_status"] == models.POSTPARTUM_STATE)
        disability_weight.loc[pregnant_or_not_pregnant] = anemia_disability_weight.loc[pregnant_or_not_pregnant]
        disability_weight.loc[maternal_disorders] = maternal_disorder_disability_weight.loc[maternal_disorders]
        disability_weight.loc[no_maternal_disorders] = 0.
        disability_weight.loc[postpartum] = 6/5 * anemia_disability_weight.loc[postpartum]
        return disability_weight

    def hemoglobin_pregnancy_adjustment(self, index: pd.Index, df: pd.DataFrame) -> pd.DataFrame:
        return df * self.correction_factors(index)

    ####################
    # Sampling helpers #
    ####################

    def _sample_child_outcomes(self, index: pd.Index):
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


