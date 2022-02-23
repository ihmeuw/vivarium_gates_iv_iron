import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.population import Mortality
from vivarium_gates_iv_iron.constants import models, data_keys, metadata
from vivarium_gates_iv_iron.constants.data_values import POSTPARTUM_DURATION_DAYS, PREPOSTPARTUM_DURATION_DAYS


class Pregnancy:
    def __init__(self):
        pass

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.clock = builder.time.clock()

        # children_born = []  # This will be input for child model

        self.columns_created = [
            'pregnancy_status',  # not_pregnant, pregnant, postpartum
            'pregnancy_outcome',    # livebirth, still birth, other
            'sex_of_child',
            'birth_weight',
            'pregnancy_state_change_date',
            'pregnancy_duration',
            'cause_of_death',
            'years_of_life_lost'
        ]

        prevalences = self.load_pregnancy_prevalence(builder)
        self.prevalence = builder.lookup.build_table(prevalences,
                                                     key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])
        conception_rate_data = builder.data.load(data_keys.PREGNANCY.INCIDENCE_RATE).fillna(0)
        conception_rate = builder.lookup.build_table(conception_rate_data,
                                                     key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])
        self.conception_rate = builder.value.register_rate_producer('conception_rate', source=conception_rate)
        conception_rate_data = conception_rate_data.set_index([col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"])

        outcome_probabilities = self.load_pregnancy_outcome_probabilities(builder)
        self.outcome_probabilities = builder.lookup.build_table(outcome_probabilities,
                                                                key_columns=['sex'],
                                                                parameter_columns=['age', 'year'])

        life_expectancy_data = builder.data.load("population.theoretical_minimum_risk_life_expectancy")
        self.life_expectancy = builder.lookup.build_table(life_expectancy_data, parameter_columns=['age'])

        all_cause_mortality_data = builder.data.load("cause.all_causes.cause_specific_mortality_rate").set_index([col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"])
        maternal_disorder_csmr = builder.data.load("cause.maternal_disorders.cause_specific_mortality_rate").set_index([col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"])
        self.background_mortality_rate = builder.lookup.build_table((all_cause_mortality_data - maternal_disorder_csmr).reset_index(),
                                                                    key_columns=['sex'],
                                                                    parameter_columns=['age', 'year'])
        maternal_disorder_incidence = builder.data.load("cause.maternal_disorders.incidence_rate").set_index([col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"])

        self.probability_maternal_deaths = builder.lookup.build_table((maternal_disorder_csmr / conception_rate_data).reset_index(),
                                                                      key_columns=['sex'],
                                                                      parameter_columns=['age', 'year'])
        self.probability_non_fatal_maternal_disorder = builder.lookup.build_table(((maternal_disorder_incidence - maternal_disorder_csmr) / conception_rate_data).reset_index(),
                                                                      key_columns=['sex'],
                                                                      parameter_columns=['age', 'year'])

        view_columns = self.columns_created + ['alive', 'exit_time', 'age', 'sex']
        self.population_view = builder.population.get_view(view_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=self.columns_created,
                                                 requires_streams=[self.name],
                                                 requires_columns=['age', 'sex'])
        builder.event.register_listener("time_step", self.on_time_step)
        # builder.event.register_listener('time_step', self.on_time_step_mortality, priority=0)

        # self.cause_specific_mortality_rate = builder.value.register_value_producer(
        #     'cause_specific_mortality_rate', source=builder.lookup.build_table(0)
        # )

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

        pregnancy_outcome_probabilities = self.outcome_probabilities(is_pregnant_idx)[list(models.PREGNANCY_OUTCOMES)]
        pregnancy_outcome.loc[is_pregnant_idx] = self.randomness.choice(is_pregnant_idx,
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
            POSTPARTUM_DURATION_DAYS * self.randomness.get_draw(pop_data.index,
                                                                additional_key='days_until_postpartum_ends'))
        postpartum_start_date = pop_data.creation_time - days_until_postpartum_ends
        pregnancy_state_change_date.loc[is_pregnant_idx] = conception_date.loc[is_pregnant_idx]
        pregnancy_state_change_date.loc[is_postpartum_idx] = postpartum_start_date.loc[is_postpartum_idx]

        # initialize columns for 'cause_of_death', 'years_of_life_lost'
        cause_of_death = pd.Series(None, index=pop_data.index)
        years_of_life_lost = pd.Series(0., index=pop_data.index)

        pop_update = pd.DataFrame({'pregnancy_status': pregnancy_status,
                                   'pregnancy_outcome': pregnancy_outcome,
                                   'sex_of_child': sex_of_child,
                                   'birth_weight': birth_weight,
                                   'pregnancy_duration': pregnancy_duration,
                                   'pregnancy_state_change_date': pregnancy_state_change_date,
                                   'cause_of_death': cause_of_death,
                                   'years_of_life_lost': years_of_life_lost
                                   })

        self.population_view.update(pop_update)

    def on_time_step_mortality(self, event: Event):

        pop = self.population_view.get(event.index, query="alive =='alive'")
        prob_df = rate_to_probability(pd.DataFrame(self.mortality_rate(pop.index)))
        prob_df['no_death'] = 1 - prob_df.sum(axis=1)
        prob_df['cause_of_death'] = self.random.choice(prob_df.index, prob_df.columns, prob_df)
        dead_pop = prob_df.query('cause_of_death != "no_death"').copy()

        if not dead_pop.empty:
            dead_pop['alive'] = pd.Series('dead', index=dead_pop.index)
            dead_pop['exit_time'] = event.time
            dead_pop['years_of_life_lost'] = self.life_expectancy(dead_pop.index)
            self.population_view.update(dead_pop[['alive', 'exit_time', 'cause_of_death', 'years_of_life_lost']])

    def on_time_step(self, event: Event):
        # TODO: figure out if simulant dies this time step
        # This cause model should be implemented such that each simulant who experiences deaths due to maternal
        # disorders also experiences an incident case of maternal disorders.
        # TODO: if they do not die, they are dying at rate of acmr - csmr
        # TODO: if they don't do either, they are alive!
        pop = self.population_view.get(event.index, query="alive =='alive'")
        not_pregnant = pop['pregnancy_status'] == models.NOT_PREGNANT_STATE

        # Make masks for subsets
        pregnancy_ends_this_step = (
                (pop['pregnancy_status'] == models.PREGNANT_STATE)
                & (event.time - pop["pregnancy_state_change_date"] > pop["pregnancy_duration"])
        )
        prepostpartum_ends_this_step = (
                                           (
                                               (pop['pregnancy_status'] == models.MATERNAL_DISORDER_STATE)
                                               | (pop['pregnancy_status'] == models.NO_MATERNAL_DISORDER_STATE)
                                               & (event.time - pop["pregnancy_state_change_date"] >
                                                  pd.Timedelta(days=PREPOSTPARTUM_DURATION_DAYS))  # One time step
                                            )
        )
        postpartum_ends_this_step = (
                (pop['pregnancy_status'] == models.POSTPARTUM_STATE)
                & (event.time - pop["pregnancy_state_change_date"] > pd.Timedelta(days=POSTPARTUM_DURATION_DAYS))
        )

        # Determine who dies
        maternal_disorder_death_draw = self.randomness.get_draw(pop.index, additional_key="maternal_disorder_death")
        would_die_due_to_maternal_disorders = maternal_disorder_death_draw < self.probability_maternal_deaths(pop.index)
        died_due_to_maternal_disorders = pregnancy_ends_this_step & would_die_due_to_maternal_disorders
        died_due_to_background_causes_index = self.randomness.filter_for_rate(pop.index, rate=self.background_mortality_rate(pop.index),
                                                                        additional_key="other_cause_death")
        died_due_to_background_causes = pd.Series(False, index=pop.index)
        died_due_to_background_causes.loc[died_due_to_background_causes_index] = True
        died_due_to_background_causes.loc[died_due_to_maternal_disorders] = False
        died_this_step = died_due_to_maternal_disorders | died_due_to_background_causes


        conception_rate = self.conception_rate(pop.index)[not_pregnant]
        pregnant_this_step = self.randomness.filter_for_rate(pop[not_pregnant].index, conception_rate,
                                                             additional_key='new_pregnancy')
        pregnancy_ends_this_step = pop.loc[(pop['pregnancy_status'] == models.PREGNANT_STATE) & (
                event.time - pop["pregnancy_state_change_date"] > pop["pregnancy_duration"])].index
        no_maternal_disorder_risk = pop.index.difference(pregnancy_ends_this_step)

        # Non-fatal maternal disorders



        p = self.outcome_probabilities(pregnant_this_step)[list(models.PREGNANCY_OUTCOMES)]
        pregnancy_outcome = self.randomness.choice(pregnant_this_step, choices=models.PREGNANCY_OUTCOMES, p=p,
                                                   additional_key='pregnancy_outcome')

        sex_of_child = self.randomness.choice(pregnant_this_step, choices=['Male', 'Female'],
                                              p=[0.5, 0.5], additional_key='sex_of_child')

        # TODO: update with birth_weight distribution
        birth_weight = 1500.0 + 1500 * self.randomness.get_draw(pregnant_this_step, additional_key='birth_weight')

        pregnancy_duration = pd.to_timedelta(9 * 28,
                                             unit='d')

        # Handle simulants going from np -> p
        new_pregnant = pd.DataFrame({'pregnancy_status': models.PREGNANT_STATE,
                                     'pregnancy_outcome': pregnancy_outcome,
                                     'sex_of_child': sex_of_child,
                                     'birth_weight': birth_weight,
                                     'pregnancy_duration': pregnancy_duration,
                                     'pregnancy_state_change_date': event.time}, index=pregnant_this_step)

        # TODO: Handle simulants going from p -> (md or nmd)
        new_postpartum = pop.loc[pregnancy_ends_this_step, self.columns_created]
        new_postpartum['pregnancy_status'] = models.POSTPARTUM_STATE
        new_postpartum['pregnancy_state_change_date'] = event.time

        # TODO: Handle simulants going from (md or nmd) -> pp


        # Handle simulants going from pp->np
        new_not_pregnant = pd.DataFrame({'pregnancy_status': models.NOT_PREGNANT_STATE,
                                         'pregnancy_outcome': models.INVALID_OUTCOME,
                                         'sex_of_child': models.INVALID_OUTCOME,
                                         'birth_weight': np.nan,
                                         'pregnancy_duration': pd.NaT,
                                         'pregnancy_state_change_date': event.time}, index=not_pregnant_this_step)

        # TODO: update all the new things (md and nmd)
        pop_update = pd.concat([new_pregnant, new_not_pregnant, new_postpartum]).sort_index()
        # TODO file bug report for pandas with pd.concat and pd.append
        pop_update['pregnancy_duration'] = pd.to_timedelta(pop_update['pregnancy_duration'])
        self.population_view.update(pop_update)

    def on_collect_metrics(self):
        # TODO: Record births, append (sex, bw, ga, birth date, maternal characteristics) tuple to list
        ...

    def on_simulation_end(self):
        # TODO: coerce list of children tuples to dataframe
        # Get output directory from configuration (can be special config key or get from general results key)
        # write to file
        ...

    def load_pregnancy_prevalence(self, builder: Builder) -> pd.DataFrame:
        index_cols = [col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != 'location']
        not_pregnant_prevalence = (builder.data.load(data_keys.PREGNANCY.NOT_PREGNANT_PREVALENCE)
                                   .fillna(0)
                                   .set_index(index_cols))
        pregnant_prevalence = (builder.data.load(data_keys.PREGNANCY.PREGNANT_PREVALENCE)
                               .fillna(0)
                               .set_index(index_cols))
        postpartum_prevalence = (builder.data.load(data_keys.PREGNANCY.POSTPARTUM_PREVALENCE)
                                 .fillna(0)
                                 .set_index(index_cols))
        maternal_disorder_prevalence = pd.Series(0., index=postpartum_prevalence.index, name=models.MATERNAL_DISORDER_STATE)
        no_maternal_disorder_prevalence = 1/6 * postpartum_prevalence
        postpartum_prevalence = 5/6 * postpartum_prevalence

        # order of prevalences must match order of PREGNANCY_MODEL_STATES
        prevalences = pd.concat([not_pregnant_prevalence, pregnant_prevalence, maternal_disorder_prevalence,
                                 no_maternal_disorder_prevalence, postpartum_prevalence], axis=1)
        prevalences.columns = list(models.PREGNANCY_MODEL_STATES)

        return prevalences.reset_index()

    def load_pregnancy_outcome_probabilities(self, builder: Builder) -> pd.DataFrame:
        pregnancy_outcome_keys = [data_keys.PREGNANCY_OUTCOMES.LIVE_BIRTH,
                                  data_keys.PREGNANCY_OUTCOMES.STILLBIRTH,
                                  data_keys.PREGNANCY_OUTCOMES.OTHER]
        outcome_probabilities = []
        index_cols = ['sex', 'age_start', 'age_end', 'year_start', 'year_end']
        for data_key, status in zip(pregnancy_outcome_keys, models.PREGNANCY_OUTCOMES[:-1]):
            p = builder.data.load(data_key)
            p = p.set_index(index_cols)['value'].rename(status).fillna(0)
            outcome_probabilities.append(p)
        outcome_probabilities = pd.concat(outcome_probabilities, axis=1)
        outcome_probabilities[models.PREGNANCY_OUTCOMES[-1]] = 1 - outcome_probabilities.sum(axis=1)

        return outcome_probabilities.reset_index()
