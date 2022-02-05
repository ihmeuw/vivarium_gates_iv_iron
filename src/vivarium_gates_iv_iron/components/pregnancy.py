import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.time import Time
from vivarium.framework.values import Pipeline
from vivarium_public_health.disease import (DiseaseState, DiseaseModel, SusceptibleState,
                                            RateTransition as RateTransition_, RecoveredState)
from vivarium_public_health.risks.base_risk import Risk
from vivarium_gates_iv_iron.constants import models, data_keys, data_values


# XXX TODO: Add DiseaseModel subclass that creates the columns we need to track


class PregnancyDisease(DiseaseModel):
    def setup(self, builder):
        """Perform this component's setup."""
        super().setup(builder)

        self.configuration_age_start = builder.configuration.population.age_start
        self.configuration_age_end = builder.configuration.population.age_end

        cause_specific_mortality_rate = self.load_cause_specific_mortality_rate_data(builder)
        self.cause_specific_mortality_rate = builder.lookup.build_table(cause_specific_mortality_rate,
                                                                        key_columns=['sex'],
                                                                        parameter_columns=['age', 'year'])
        builder.value.register_value_modifier('cause_specific_mortality_rate',
                                              self.adjust_cause_specific_mortality_rate,
                                              requires_columns=['age', 'sex'])

        self.population_view = builder.population.get_view(['age', 'sex', self.state_column])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.state_column],
                                                 requires_columns=['age', 'sex'],
                                                 requires_streams=[f'{self.state_column}_initial_states'])
        self.randomness = builder.randomness.get_stream(f'{self.state_column}_initial_states')

        builder.event.register_listener('time_step', self.on_time_step)
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)


    #TODO create columns
    #TODO on initialize_simulants populate the columns

    def on_initialize_simulants(self, pop_data):
        population = self.population_view.subview(['age', 'sex']).get(pop_data.index)

        assert self.initial_state in {s.state_id for s in self.states}

        # Assumption being made there is no fertility in this model
        state_names, weights_bins = self.get_state_weights(pop_data.index, "prevalence")

        if state_names and not population.empty:
            # only do this if there are states in the model that supply prevalence data
            population['sex_id'] = population.sex.apply({'Male': 1, 'Female': 2}.get)

            condition_column = self.assign_initial_status_to_simulants(population, state_names, weights_bins,
                                                                       self.randomness.get_draw(population.index))

            condition_column = condition_column.rename(columns={'condition_state': self.state_column})
        else:
            condition_column = pd.Series(self.initial_state, index=population.index, name=self.state_column)

        #TODO initialize values for new columns (child sex, pregnancy outcome, gestational age, birth weight)
        #TODO create all columns with index and default values for non-pregnant women
        #TODO if woman is pregnant, find sex of child
        #    -get propensity for each simulant by calling randomness stream (add additional key for child sex)

        #TODO create risk component for child sex
        #    -generate pregnancy_pipeline for sex and another pipeline for outcome, get_current_exposure in risk pipeline

        #TODO create risk component for pregnancy outcome

        #TODO for outcomes of livebirth, find gestational age and birth weight
        #    -apply LBWGG risk

        self.population_view.update(condition_column)


class ChildSex(Risk):

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

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _get_current_exposure(self, index: pd.Index) -> pd.Series:
        propensity = self.propensity(index)
        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)


class PregnancyOutcome(Risk):
    pass


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