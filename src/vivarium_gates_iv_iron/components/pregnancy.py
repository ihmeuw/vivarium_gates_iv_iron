import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.components.hemoglobin import Hemoglobin
from vivarium_gates_iv_iron.constants import models, data_keys
from vivarium_gates_iv_iron.constants.data_values import (
    DURATIONS,
    HEMOGLOBIN_DISTRIBUTION_PARAMETERS,
    MATERNAL_HEMORRHAGE_SEVERITY_PROBABILITY,
)
from vivarium_gates_iv_iron.utilities import (
    create_draw,
    get_norm_from_quantiles,
    get_random_variable,
    get_truncnorm_from_quantiles,
)


class Pregnancy:
    def __init__(self):
        self.hemoglobin_distribution = Hemoglobin()

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    @property
    def sub_components(self):
        return [self.hemoglobin_distribution]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.randomness = builder.randomness.get_stream(self.name)

        self.columns_created = [
            'pregnancy_status',  # not_pregnant, pregnant, postpartum
            'pregnancy_outcome',  # livebirth, still birth, other
            'sex_of_child',
            'birth_weight',
            'pregnancy_state_change_date',
            'pregnancy_duration',
            'cause_of_death',
            'years_of_life_lost',
            'maternal_hemorrhage',
        ]

        self.prevalence = builder.lookup.build_table(
            builder.data.load(data_keys.PREGNANCY.PREVALENCE),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        conception_rate_table = builder.lookup.build_table(
            builder.data.load(data_keys.PREGNANCY.INCIDENCE_RATE),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        self.conception_rate = builder.value.register_rate_producer(
            'conception_rate',
            source=conception_rate_table,
        )

        self.outcome_probabilities = builder.lookup.build_table(
            builder.data.load(data_keys.PREGNANCY.CHILD_OUTCOME_PROBABILITIES),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        self.life_expectancy = builder.lookup.build_table(
            builder.data.load(data_keys.POPULATION.TMRLE),
            parameter_columns=['age'],
        )

        all_cause_mortality_data = builder.data.load(data_keys.POPULATION.ACMR)
        maternal_disorder_csmr = builder.data.load(data_keys.MATERNAL_DISORDERS.CSMR)
        background_mortality_rate_table = builder.lookup.build_table(
            (all_cause_mortality_data - maternal_disorder_csmr).reset_index(),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        self.background_mortality_rate = builder.value.register_rate_producer(
            'background_mortality_rate',
            source=background_mortality_rate_table,
        )

        self.probability_fatal_maternal_disorder = builder.lookup.build_table(
            builder.data.load(data_keys.PREGNANCY.PROBABILITY_FATAL_MATERNAL_DISORDER),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        self.probability_non_fatal_maternal_disorder = builder.lookup.build_table(
            builder.data.load(data_keys.PREGNANCY.PROBABILITY_NONFATAL_MATERNAL_DISORDER),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )
        self.probability_maternal_hemorrhage = builder.lookup.build_table(
            builder.data.load(data_keys.PREGNANCY.PROBABILITY_MATERNAL_HEMORRHAGE),
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

        # Get value for the probability of moderate maternal hemorrhage.
        # The probability of severe maternal hemorrhage is 1 minus that probability.
        self.maternal_hemorrhage_severity = create_draw(self.draw,
                                                        MATERNAL_HEMORRHAGE_SEVERITY_PROBABILITY,
                                                        "maternal_hemorrhage_severity",
                                                        self.location,
                                                        distribution_function=get_truncnorm_from_quantiles)

        builder.value.register_value_modifier("hemoglobin.exposure_parameters",
                                              self.hemoglobin_pregnancy_adjustment,
                                              requires_columns=["pregnancy_status"])

        self.correction_factors = self.sample_correction_factors(builder)

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
        pregnancy_state_probabilities = self.prevalence(pop_data.index)[list(models.PREGNANCY_MODEL_STATES)]
        probs_all_zero = (pregnancy_state_probabilities.sum(axis=1) == 0).reset_index(drop=True)

        ages = self.population_view.subview(['age']).get(pop_data.index)
        # TODO: This code is to ensure under 10 y.o. simulants have a prevalence of not_pregnant of 1. This should
        # probably be done in the artifact itself to avoid special casing.
        is_under_ten = ages.age < 10
        assert (is_under_ten.equals(probs_all_zero))
        pregnancy_state_probabilities.loc[is_under_ten, 'not_pregnant'] = 1
        pregnancy_status = self.randomness.choice(pop_data.index, choices=models.PREGNANCY_MODEL_STATES,
                                                  p=pregnancy_state_probabilities,
                                                  additional_key='pregnancy_status')
        pregnancy_outcome = pd.Series(models.INVALID_OUTCOME, index=pop_data.index)
        is_pregnant_idx = pop_data.index[pregnancy_status == models.PREGNANT_STATE]
        is_postpartum_idx = pop_data.index[pregnancy_status == models.POSTPARTUM_STATE]
        is_prepostpartum_idx = pop_data.index[((pregnancy_status == models.NO_MATERNAL_DISORDER_STATE)
                                               | (pregnancy_status == models.MATERNAL_DISORDER_STATE))]

        # TODO refactor these calls...
        pregnancy_outcome_probabilities = self.outcome_probabilities(is_pregnant_idx)[list(models.PREGNANCY_OUTCOMES)]
        pregnancy_outcome.loc[is_pregnant_idx] = self.randomness.choice(is_pregnant_idx,
                                                                        choices=models.PREGNANCY_OUTCOMES,
                                                                        p=pregnancy_outcome_probabilities,
                                                                        additional_key='pregnancy_outcome')

        pregnancy_outcome_probabilities = self.outcome_probabilities(is_postpartum_idx)[list(models.PREGNANCY_OUTCOMES)]
        pregnancy_outcome.loc[is_postpartum_idx] = self.randomness.choice(is_postpartum_idx,
                                                                        choices=models.PREGNANCY_OUTCOMES,
                                                                        p=pregnancy_outcome_probabilities,
                                                                        additional_key='pregnancy_outcome')

        pregnancy_outcome_probabilities = self.outcome_probabilities(is_prepostpartum_idx)[list(models.PREGNANCY_OUTCOMES)]
        pregnancy_outcome.loc[is_prepostpartum_idx] = self.randomness.choice(is_prepostpartum_idx,
                                                                        choices=models.PREGNANCY_OUTCOMES,
                                                                        p=pregnancy_outcome_probabilities,
                                                                        additional_key='pregnancy_outcome')

        sex_of_child = pd.Series(models.INVALID_OUTCOME, index=pop_data.index)
        # TODO: update sex_of_child distribution
        sex_of_child.loc[is_pregnant_idx] = self.randomness.choice(is_pregnant_idx, choices=['Male', 'Female'],
                                                                   p=[0.5, 0.5], additional_key='sex_of_child')

        birth_weight = pd.Series(np.nan, index=pop_data.index)
        # TODO implement LBWSG on next line for sampling
        birth_weight.loc[is_pregnant_idx] = 1500.0 + 1500 * self.randomness.get_draw(is_pregnant_idx,
                                                                                     additional_key='birth_weight')

        pregnancy_duration = pd.Series(pd.NaT, index=pop_data.index)
        pregnancy_duration.loc[is_pregnant_idx] = pd.to_timedelta(9 * 28,
                                                                  unit='d')

        pregnancy_state_change_date = pd.Series(pd.NaT, index=pop_data.index)
        days_until_pregnancy_ends = pregnancy_duration * self.randomness.get_draw(pop_data.index,
                                                                                  additional_key='conception_date')
        conception_date = pop_data.creation_time - days_until_pregnancy_ends

        days_until_postpartum_ends = pd.to_timedelta(
             7 * DURATIONS.POSTPARTUM * self.randomness.get_draw(pop_data.index,
                                                                 additional_key='days_until_postpartum_ends'))
        postpartum_start_date = pop_data.creation_time - days_until_postpartum_ends
        pregnancy_state_change_date.loc[is_pregnant_idx] = conception_date.loc[is_pregnant_idx]
        pregnancy_state_change_date.loc[is_postpartum_idx] = postpartum_start_date.loc[is_postpartum_idx]
        pregnancy_state_change_date.loc[is_prepostpartum_idx] = (pop_data.creation_time
                                                                 - pd.Timedelta(days=7 * DURATIONS.PREPOSTPARTUM))

        # initialize columns for 'cause_of_death', 'years_of_life_lost'
        cause_of_death = pd.Series("not_dead", index=pop_data.index, dtype="string")
        years_of_life_lost = pd.Series(0., index=pop_data.index)

        # Initialize columns for maternal hemorrhage, hemoglobin, anemia
        maternal_hemorrhage = pd.Series(models.NOT_MATERNAL_HEMORRHAGE_STATE, index=pop_data.index)

        pop_update = pd.DataFrame({'pregnancy_status': pregnancy_status,
                                   'pregnancy_outcome': pregnancy_outcome,
                                   'sex_of_child': sex_of_child,
                                   'birth_weight': birth_weight,
                                   'pregnancy_duration': pregnancy_duration,
                                   'pregnancy_state_change_date': pregnancy_state_change_date,
                                   'cause_of_death': cause_of_death,
                                   'years_of_life_lost': years_of_life_lost,
                                   'maternal_hemorrhage': maternal_hemorrhage,
                                   })

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

        # Determine who dies
        maternal_disorder_death_draw = self.randomness.get_draw(pop.index, additional_key="maternal_disorder_death")
        would_die_due_to_maternal_disorders = maternal_disorder_death_draw < self.probability_maternal_deaths(pop.index)
        died_due_to_maternal_disorders = pregnancy_ends_this_step & would_die_due_to_maternal_disorders
        died_due_to_background_causes_index = self.randomness.filter_for_rate(pop.index,
                                                                              rate=self.background_mortality_rate(
                                                                                  pop.index),
                                                                              additional_key="other_cause_death")
        died_due_to_background_causes = pd.Series(False, index=pop.index)
        died_due_to_background_causes.loc[died_due_to_background_causes_index] = True
        died_due_to_background_causes.loc[died_due_to_maternal_disorders] = False
        died_this_step = died_due_to_maternal_disorders | died_due_to_background_causes

        pop.loc[died_this_step, "alive"] = "dead"
        pop.loc[died_this_step, "exit_time"] = event.time
        pop.loc[died_this_step, "years_of_life_lost"] = self.life_expectancy(pop.loc[died_this_step].index)
        pop.loc[died_due_to_maternal_disorders, "cause_of_death"] = "maternal_disorders"
        pop.loc[died_due_to_background_causes, "cause_of_death"] = "other_causes"

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
        pop = self.population_view.get(index)
        for state in models.PREGNANCY_MODEL_STATES:
            state_index = pop[pop["pregnancy_status"] == state].index
            df.loc[state_index, "mean"] *= self.correction_factors[state][0]
            df.loc[state_index, "stddev"] *= self.correction_factors[state][1]
        return df

    def sample_correction_factors(self, builder: Builder):
        seed = builder.configuration.randomness.random_seed
        draw = builder.configuration.input_data.input_draw_number

        not_pregnant_mean_cf = get_random_variable(draw, seed, get_norm_from_quantiles(*HEMOGLOBIN_DISTRIBUTION_PARAMETERS.NO_PREGNANCY_MEAN_ADJUSTMENT_FACTOR))
        not_pregnant_sd_cf = get_random_variable(draw, seed, get_norm_from_quantiles(*HEMOGLOBIN_DISTRIBUTION_PARAMETERS.NO_PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR))
        pregnant_mean_cf = get_random_variable(draw, seed, get_norm_from_quantiles(*HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_MEAN_ADJUSTMENT_FACTOR))
        pregnant_sd_cf = get_random_variable(draw, seed, get_norm_from_quantiles(
            *HEMOGLOBIN_DISTRIBUTION_PARAMETERS.PREGNANCY_STANDARD_DEVIATION_ADJUSTMENT_FACTOR))
        correction_factors = {models.NOT_PREGNANT_STATE: (not_pregnant_mean_cf, not_pregnant_sd_cf)}
        for state in models.PREGNANCY_MODEL_STATES[1:]:
            correction_factors[state] = (pregnant_mean_cf, pregnant_sd_cf)
        return correction_factors

