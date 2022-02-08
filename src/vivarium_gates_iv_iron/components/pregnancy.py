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
    def __init__(self):
        pass

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    def setup(self, builder: Builder):
        self.population_view = self._get_population_view(builder)

        children_born = []  # This will be input for child model

        columns_created = [
            'pregnancy_status',  # not_pregnant, pregnant, postpartum
            'pregnancy_outcome',
            'child_sex',
            'birth_weight',
            'conception_date',
            'pregnancy_duration',
        ]

        outcomes = ["stillbirth", "live_birth", "other"]

        # pipelines = [
        #     'pregnancy_outcome',
        #     'birth_weight_shift',
        # ]

        index_cols = [col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != 'location']
        pregnant_prevalence = (builder.data.load(data_keys.PREGNANCY.PREVALENCE)
                               .dropna()
                               .reset_index()
                               .set_index(index_cols)
                               .drop('index', axis=1))
        postpartum_prevalence = pregnant_prevalence * 6 / 40
        not_pregnant_prevalence = 1 - (postpartum_prevalence + pregnant_prevalence)
        prevalences = pd.concat([not_pregnant_prevalence, pregnant_prevalence, postpartum_prevalence], axis=1)
        prevalences.columns = ['not_pregnant', 'pregnant', 'postpartum']
        self.prevalence = builder.lookup.build_table(prevalences.reset_index(),
                                                     key_columns=['sex'],
                                                     parameter_columns=['age', 'year'])
        builder.population.initializes_simulants(self.on_initialize_simulants)
        self.randomness = builder.randomness.get_stream(self.name)

    def _get_population_view(self, builder: Builder) -> PopulationView:
        # just pregnancy status for now
        return builder.population.get_view(['tracked', 'pregnancy_status'])

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # TODO sample pregnant | age, year, assign pregnancy status
        pregnancy_status = self.randomness.choice(pop_data.index, choices=['np','pp','p'], p=self.prevalence(pop_data.index))
        pregnancy_status = pd.Series(pregnancy_status, name='pregnancy_status')
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

    def determine_pregnancy_state(self):
        #TODO: give simulant outcome: do they transition or stay in current state?
        pass

    def get_pregnancy_outcome(self):
        #TODO: get result of pregnancy [stillbirth, live_birth, other]
        pass

    def get_sex_of_child(self):
        #TODO: use location probabilities to get sex of child IF determine_prenancy_ouotcome is stillbirth or live_birth
        pass

    def get_pregnancy_data(self):
        #TODO: get gestational age - do we also need birth weight?
        #TODO: get pregnancy duration
        #TODO: get conception date
        pass

    def clean_newborn_data(self):
        #TODO: clean/prep data to be input for child model
        pass