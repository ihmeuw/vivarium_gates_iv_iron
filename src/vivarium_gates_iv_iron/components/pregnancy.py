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
    PREGNANCY_OUTCOMES = ("stillbirth", "live_birth", "other")

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
            # 'pregnancy_outcome',
            # 'child_sex',
            # 'birth_weight',
            # 'conception_date',
            # 'pregnancy_duration',
        ]

        prevalences = self.load_pregnancy_prevalence(builder)
        self.prevalence = builder.lookup.build_table(prevalences,
                                                     key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])

        self.population_view = builder.population.get_view(columns_created)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_streams=[self.name])


    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # TODO sample pregnant | age, year, assign pregnancy status
        p = self.prevalence(pop_data.index)[list(self.PREGNANCY_STATUSES)]
        pregnancy_status = self.randomness.choice(pop_data.index, choices=self.PREGNANCY_STATUSES, p=p)
        self.population_view.update(pregnancy_status)
        # TODO sample pregnancy outcome | pregnancy status
        # is_pregnant = self.population_view.subview(['pregnancy_status']).get(pop_data.index).squeeze(axis=1) == 'p'
        # outcome is default if pregnancy status is not pregnant or postpartum
        # outcome is binned into 3 possible outcomes if status is pregnant
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
                               .dropna()
                               .reset_index()
                               .set_index(index_cols)
                               .drop('index', axis=1))
        postpartum_prevalence = pregnant_prevalence * 6 / 40
        not_pregnant_prevalence = 1 - (postpartum_prevalence + pregnant_prevalence)
        # order of prevalences must match order of PREGNANCY_STATUSES
        prevalences = pd.concat([not_pregnant_prevalence, pregnant_prevalence, postpartum_prevalence], axis=1)
        prevalences.columns = list(self.PREGNANCY_STATUSES)

        return prevalences.reset_index()