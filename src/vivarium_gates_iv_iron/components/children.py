import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_gates_iv_iron.constants import models


class NewChildren:

    @property
    def name(self):
        return 'child_status'

    @property
    def columns_created(self):
        return ['sex_of_child', 'birth_weight']

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

    def empty(self, index: pd.Index):
        return pd.DataFrame({
            'sex_of_child': models.INVALID_OUTCOME,
            'birth_weight': np.nan,
        }, index=index)

    def __call__(self, index: pd.Index):
        sex_of_child = self.randomness.choice(
            index,
            choices=['Male', 'Female'],
            additional_key='sex_of_child',
        )
        # TODO implement LBWSG on next line for sampling
        draw = self.randomness.get_draw(index, additional_key='birth_weight')
        birth_weight = 1500. * (1 + draw)

        return pd.DataFrame({
            'sex_of_child': sex_of_child,
            'birth_weight': birth_weight,
        }, index=index)