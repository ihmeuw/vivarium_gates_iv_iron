import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_gates_iv_iron.constants import (
    data_keys,
    data_values,
    models,
)


class MaternalDisability:

    @property
    def name(self):
        return 'maternal_disability'

    def setup(self, builder: Builder):
        self.step_size = builder.time.step_size()
        self.ylds_per_maternal_disorder = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.YLDS),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        self.anemia_disability = builder.value.get_value('anemia.disability_weight')
        builder.value.register_value_modifier(
            "disability_weight",
            self.accrue_disability,
            requires_columns=["alive", "pregnancy_status"],
        )

        self.population_view = builder.population.get_view(['alive', 'pregnancy_status'])

    def accrue_disability(self, index: pd.Index):
        anemia_disability_weight = self.anemia_disability(index)
        maternal_disorder_ylds = self.ylds_per_maternal_disorder(index)
        maternal_disorder_disability_weight = (
            maternal_disorder_ylds * 365 / self.step_size().days
        )

        postpartum_scalar = (
            data_values.DURATIONS.POSTPARTUM
            / (data_values.DURATIONS.POSTPARTUM + data_values.DURATIONS.PREPOSTPARTUM)
        )
        dw_map = {
            models.NOT_PREGNANT_STATE: anemia_disability_weight,
            models.PREGNANT_STATE: anemia_disability_weight,
            models.NO_MATERNAL_DISORDER_STATE: 0. * maternal_disorder_disability_weight,
            models.MATERNAL_DISORDER_STATE: maternal_disorder_disability_weight,
            models.POSTPARTUM_STATE: postpartum_scalar * anemia_disability_weight
        }

        pop = self.population_view.get(index)
        alive = pop["alive"] == "alive"
        disability_weight = pd.Series(np.nan, index=index)
        for state, dw in dw_map.items():
            in_state = alive & (pop['pregnancy_status'] == state)
            disability_weight[in_state] = dw.loc[in_state]

        return disability_weight
