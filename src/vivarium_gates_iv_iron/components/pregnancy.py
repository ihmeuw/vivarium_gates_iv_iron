from vivarium_public_health.disease import (DiseaseState, DiseaseModel, SusceptibleState,
                                            RateTransition as RateTransition_, RecoveredState)

from vivarium_gates_iv_iron.constants import models, data_keys, data_values


# XXX TODO: Add DiseaseModel subclass that creates the columns we need to track


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
        # side_effect_function=func
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