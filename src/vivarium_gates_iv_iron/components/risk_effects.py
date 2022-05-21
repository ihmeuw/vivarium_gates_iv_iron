from typing import Dict, TYPE_CHECKING


import numpy as np
import pandas as pd
from scipy import stats

from vivarium_public_health.risks.effect import RiskEffect
from vivarium_public_health.risks import Risk

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.randomness import get_hash
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.constants import data_keys, data_values
from vivarium_gates_iv_iron.data.sampling import get_lognorm_from_quantiles

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


class HemoglobinRiskEffects:
    # Hemoglobin effect on MH incidence
    # MH effect on hemoglobin
    # Hemoglobin effect on maternal disorders

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.draw = builder.configuration.input_data.input_draw_number
        self.randomness = builder.randomness.get_stream(self.name)
        required_columns = [
            "age",
            "sex",
            # TODO: Add more here
        ]
        self.required_columns = required_columns
        created_columns = []  #[list(data_values.RISKS)]


        #         self.ylds_per_maternal_disorder = builder.lookup.build_table(
        #             builder.data.load(data_keys.MATERNAL_DISORDERS.YLDS),
        #             key_columns=['sex'],
        #             parameter_columns=['age', 'year'],
        #         )

        # Get RR, PAF, etc.
        # TODO: change to lookup
        self.p_hgb_70 = builder.data.load(data_keys.HEMOGLOBIN.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70)
        self.rr_maternal_hemorrhage_attributable_to_hemoglobin = get_rr(
            get_lognorm_from_quantiles(*data_values.RR_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN),
            f"rr_maternal_hemorrhage_attributable_to_hemoglobin_draw_{self.draw}"
        )
        self.paf_maternal_hemorrhage_attributable_to_hemoglobin = pd.Series()



        self.population_view = builder.population.get_view(required_columns + created_columns)

        # builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=created_columns,
        #                                          requires_columns=required_columns, requires_streams=[self.name])

        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)


    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        self.paf_maternal_hemorrhage_attributable_to_hemoglobin = calculate_paf(
            self.p_hgb_70(pop_data.index), self.rr_maternal_hemorrhage_attributable_to_hemoglobin)
        # pop = self.population_view.subview(self.required_columns).get(pop_data.index)
        # pop_update = self.set_values_on_diagnosis(pop)
        # self.population_view.update(pop_update)


    def on_time_step_cleanup(self, event: 'Event'):
        pass
        # pop = self.population_view.get(event.index)
        # newly_with_condition = pop[f'{models.MULTIPLE_MYELOMA_1_STATE_NAME}_event_time'] == event.time
        # pop = pop.loc[newly_with_condition]
        # if not pop.empty:
        #     pop_update = self.set_values_on_diagnosis(pop)
        #     self.population_view.update(pop_update)


def get_rr(distribution: stats.lognorm, seed: str) -> float:
    np.random.seed(get_hash(seed))
    return distribution.rvs()


def calculate_paf(proportion: pd.Series, rr: float) -> pd.Series:
    return (
        (rr * proportion + (1 - proportion) - 1)
        / (rr * proportion + (1 - proportion))
    )