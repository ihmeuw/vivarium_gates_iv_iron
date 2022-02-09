import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.time import Time
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import (DiseaseState, DiseaseModel, SusceptibleState,
                                            RateTransition as RateTransition_, RecoveredState)
from vivarium_public_health.risks.base_risk import Risk
from vivarium_gates_iv_iron.constants import models, data_keys, data_values, metadata


class Pregnancy:
    PREGNANCY_STATUSES = ('not_pregnant', 'pregnant', 'postpartum')
    PREGNANCY_OUTCOMES = ("live_birth", "stillbirth", "other", "invalid")

    def __init__(self):
        pass

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

        #children_born = []  # This will be input for child model

        columns_created = [
            'pregnancy_status',  # not_pregnant, pregnant, postpartum
            'pregnancy_outcome',
            'sex_of_child',
            # 'birth_weight',
            # 'conception_date',
            # 'pregnancy_duration',
        ]

        prevalences = self.load_pregnancy_prevalence(builder)
        self.prevalence = builder.lookup.build_table(prevalences,
                                                     key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])
        outcome_probabilities = self.load_pregnancy_outcome_probabilities(builder)
        self.outcome_probabilities = builder.lookup.build_table(outcome_probabilities,
                                                     key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])
        #TODO remove age and sex colums when done debugging
        self.population_view = builder.population.get_view(columns_created + ['age', 'sex'])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_streams=[self.name],
                                                 requires_columns=['age', 'sex'])


    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # TODO sample pregnant | age, year, assign pregnancy status
        p = self.prevalence(pop_data.index)[list(self.PREGNANCY_STATUSES)]
        pregnancy_status = self.randomness.choice(pop_data.index, choices=self.PREGNANCY_STATUSES, p=p, additional_key='pregnancy_status')
        pregnancy_outcome = pd.Series('invalid', index=pop_data.index)
        is_pregnant_idx = pop_data.index[pregnancy_status == 'pregnant']
        if not is_pregnant_idx.empty:
            p = self.outcome_probabilities(is_pregnant_idx)[list(self.PREGNANCY_OUTCOMES)]
            pregnancy_outcome.loc[is_pregnant_idx] = self.randomness.choice(is_pregnant_idx, choices=self.PREGNANCY_OUTCOMES, p=p, additional_key='pregnancy_outcome')

        sex_of_child = pd.Series('invalid', index=pop_data.index)
        sex_of_child.loc[is_pregnant_idx] = self.randomness.choice(is_pregnant_idx, choices=['Male', 'Female'], p=[0.5, 0.5], additional_key='sex_of_child')

        pop_update = pd.DataFrame({'pregnancy_status': pregnancy_status,
                                   'pregnancy_outcome': pregnancy_outcome,
                                   'sex_of_child': sex_of_child})
        self.population_view.update(pop_update)

        # TODO sample child sex | pregnancy outcome

        # TODO sample gestational_age | pregnancy_status, child_sex, pregnancy_outcome) assign pregnancy duration
        # TODO conception_date | gestational_age) (uniformly between now and gestational age

    def on_time_step(self):
        # if not pregnant,
        # do you get pregnant, if so, sample gestational age, set pregnancy status and gestational age and conception date

        # if pregnant
        # do you move to postpartum (is conception date + pregnancy_duration > t)

        # if postpartum
        # do you move to not pregnant
        ...

    def on_collect_metrics(self):
        # Record births, append (sex, bw, ga, birth date, maternal characteristics) tuple to list
        ...

    def on_simulation_end(self):
        # coerce list of children tuples to dataframe
        # Get output directory from configuration (can be special config key or get from general results key)
        # write to file
        ...


    def load_pregnancy_prevalence(self, builder: Builder) -> pd.DataFrame:

        index_cols = [col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != 'location']
        pregnant_prevalence = (builder.data.load(data_keys.PREGNANCY.PREVALENCE)
                               .fillna(0)
                               .set_index(index_cols))
        postpartum_prevalence = pregnant_prevalence * 6 / 40
        not_pregnant_prevalence = 1 - (postpartum_prevalence + pregnant_prevalence)
        # order of prevalences must match order of PREGNANCY_STATUSES
        prevalences = pd.concat([not_pregnant_prevalence, pregnant_prevalence, postpartum_prevalence], axis=1)
        prevalences.columns = list(self.PREGNANCY_STATUSES)

        return prevalences.reset_index()

    def load_pregnancy_outcome_probabilities(self, builder: Builder) -> pd.DataFrame:

        pregnancy_outcome_keys = [data_keys.PREGNANCY_OUTCOMES.LIVE_BIRTH,
                                  data_keys.PREGNANCY_OUTCOMES.STILLBIRTH,
                                  data_keys.PREGNANCY_OUTCOMES.OTHER]
        outcome_probabilities = []
        index_cols = ['sex', 'age_start', 'age_end', 'year_start', 'year_end']
        for data_key, status in zip(pregnancy_outcome_keys, self.PREGNANCY_OUTCOMES[:-1]):
            p = builder.data.load(data_key)
            p = p.set_index(index_cols)['value'].rename(status).fillna(0)
            outcome_probabilities.append(p)
        outcome_probabilities = pd.concat(outcome_probabilities, axis=1)
        outcome_probabilities[self.PREGNANCY_OUTCOMES[-1]] = 1 - outcome_probabilities.sum(axis=1)

        return outcome_probabilities.reset_index()
