import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.time import Time
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import (DiseaseState, DiseaseModel, SusceptibleState,
                                            RateTransition as RateTransition_, RecoveredState)
from vivarium_public_health.risks.base_risk import Risk
from vivarium_gates_iv_iron.constants import models, data_keys, data_values


class Pregnancy:
    def __init__(self):

    @property
    def name(self):
        return models.PREGNANCY_MODEL_NAME

    def setup(self, builder: Builder):
        children_born = []  # This will be input for child model

        columns_created = [
            'pregnancy_status',  # not_pregnant, pregnant, postpartum
            'pregnancy_outcome',
            'child_sex',
            'birth_weight',
            'conception_date',
            'pregnancy_duration',
        ]

        # pipelines = [
        #     'pregnancy_outcome',
        #     'birth_weight_shift',
        # ]

    def on_initialize_simulants(self):
        # TODO sample pregnant | age, year, assign pregnancy status
        # TODO sample pregnancy outcome | pregnancy status
        # TODO sample child sex | pregnancy outcome
        # TODO sample gestational_age | pregnancy_status, child_sex, pregnancy_outcome) assign pregnancy duration
        # TODO conception_date | gestational_age) (uniformly between now and gestational age
        pass

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

    def clean_newborn_data(self):
        #TODO: clean/prep data to be input for child model
        pass

class LBWSGDistribution:
    # Wrap around core pieces of LBWSG risk component

    def setup(self, builder):
        pass

    def sample(self, list_of_sexes) -> List[Tuple(float, float)]:
        # take an int n_samples, give back a list of (bw, ga) tuples of length n_samples
        pass


class SexOfChild(Risk):
    # Risk effect to deterine child sex
    def setup(self, builder: Builder) -> None:
        self.randomness = self._get_randomness_stream(builder)
        self.propensity = self._get_sex_propensity_pipeline(builder)
        self.exposure = self._get_exposure_pipeline(builder)
        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)

    def _get_sex_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=self.randomness.get_draw,
            requires_columns=[self.propensity_column_name]
        )


class PregnancyOutcome(Risk):
    #TODO fix to get pregnancy outcome off of probabilities
    def setup(self, builder: Builder) -> None:
        self.randomness = self._get_randomness_stream(builder)
        self.propensity = self._get_outcome_propensity_pipeline(builder)
        self.exposure = self._get_exposure_pipeline(builder)
        self.population_view = self._get_population_view(builder)

        self._register_simulant_initializer(builder)

    def _get_outcome_propensity_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.propensity_pipeline_name,
            source=self.randomness.get_draw,
            requires_columns=[self.propensity_column_name]
        )


def determine_pregnancy_outcomes(index: pd.Index, event_time: 'Time') -> None:
    # This is transition side effect
    #TODO determine sex of child
    #TODO determine whether it is still_birth, full term, etc
    #TODO determine birth weight and gestational age
    # Do we want to record time of conception?
    pass

def Pregnancy():
    not_pregnant = SusceptibleState(models.PREGNANCY_MODEL_NAME)
    pregnant = DiseaseState(
        models.PREGNANT_STATE,
        get_data_functions={
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            # TODO: update with gestational age dwell time
            'dwell_time': lambda *_: '9 months'
            # XXX TODO: define side effect function to determine pregnancy attrs.
        },
        side_effect_function=determine_pregnancy_outcomes
    )
    postpartum = DiseaseState(
        models.POSTPARTUM_STATE,
        get_data_functions={
            'disability_weight': lambda *_: 0,
            'excess_mortality_rate': lambda *_: 0,
            'dwell_time': lambda *_: '6 weeks'
        },
    )

    not_pregnant.allow_self_transitions()
    not_pregnant.add_transition(
        pregnant,
        source_data_type='rate',
        get_data_functions={
            'incidence_rate': lambda _, builder: builder.data.load(data_keys.PREGNANCY.INCIDENCE_RATE)
        }
    )

    pregnant.allow_self_transitions()
    pregnant.add_transition(
        postpartum,
        #source_data_type='rate',
        # TODO: consider incidences of different pregnancy outcomes
        #get_data_functions={
        #    'incidence_rate': lambda _, builder: builder.data.load(data_keys.PREGNANCY.INCIDENCE_RATE)
        #}
    )

    postpartum.allow_self_transitions()
    postpartum.add_transition(
        not_pregnant,
        # source_data_type='rate',
        # get_data_functions={
        #     'incidence_rate': lambda _, builder: builder.data.load(data_keys.PREGNANCY.INCIDENCE_RATE)
        # }
    )
    return DiseaseModel(
        models.PREGNANCY_MODEL_NAME, states=[not_pregnant, pregnant, postpartum], initial_state=not_pregnant)