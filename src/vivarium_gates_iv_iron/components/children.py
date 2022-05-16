from typing import Tuple

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder

from vivarium_gates_iv_iron.constants import (
    data_keys,
    models,
)


class NewChildren:

    def __init__(self):
        self.lbwsg = LBWSGDistribution()

    @property
    def name(self):
        return 'child_status'

    @property
    def sub_components(self):
        return [self.lbwsg]

    @property
    def columns_created(self):
        return ['sex_of_child', 'birth_weight', 'gestational_age']

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

    def empty(self, index: pd.Index):
        return pd.DataFrame({
            'sex_of_child': models.INVALID_OUTCOME,
            'birth_weight': np.nan,
            'gestational_age': np.nan,
        }, index=index)

    def __call__(self, index: pd.Index):
        sex_of_child = self.randomness.choice(
            index,
            choices=['Male', 'Female'],
            additional_key='sex_of_child',
        )
        lbwsg = self.lbwsg(sex_of_child)
        return pd.DataFrame({
            'sex_of_child': sex_of_child,
            'birth_weight': lbwsg['birth_weight'],
            'gestational_age': lbwsg['gestational_age'],
        }, index=index)


class LBWSGDistribution:

    @property
    def name(self):
        return 'lbwsg_distribution'

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.exposure = builder.data.load(data_keys.LBWSG.EXPOSURE).set_index('sex')
        self.category_intervals = self._get_category_intervals(builder)

    def __call__(self, newborn_sex: pd.Series):
        categorical_exposure = self._sample_categorical_exposure(newborn_sex)
        continuous_exposure = self._sample_continuous_exposure(categorical_exposure)
        return continuous_exposure

    ############
    # Sampling #
    ############

    def _sample_categorical_exposure(self, newborn_sex: pd.Series):
        categorical_exposures = []
        for sex in newborn_sex.unique():
            group_data = newborn_sex[newborn_sex == sex]
            sex_exposure = self.exposure.loc[sex]
            categorical_exposures.append(self.randomness.choice(
                group_data.index,
                choices=sex_exposure.parameter.tolist(),
                p=sex_exposure.value.tolist(),
                additional_key='categorical_exposure',
            ))
        categorical_exposures = pd.concat(categorical_exposures).sort_index()
        return categorical_exposures

    def _sample_continuous_exposure(self, categorical_exposure: pd.Series):
        intervals = self.category_intervals.loc[categorical_exposure]
        intervals.index = categorical_exposure.index
        exposures = []
        for axis in ['birth_weight', 'gestational_age']:
            draw = self.randomness.get_draw(categorical_exposure.index, additional_key=axis)
            lower, upper = intervals[f'{axis}_lower'], intervals[f'{axis}_upper']
            exposures.append((lower + (upper - lower) * draw).rename(axis))
        return pd.concat(exposures, axis=1)

    ################
    # Data loading #
    ################

    def _get_category_intervals(self, builder: Builder):
        categories = builder.data.load(data_keys.LBWSG.CATEGORIES)
        category_intervals = pd.DataFrame(
            data=[(category, *self._parse_description(description))
                  for category, description in categories.items()],
            columns=['category',
                     'birth_weight_lower', 'birth_weight_upper',
                     'gestational_age_lower', 'gestational_age_upper'],
        ).set_index('category')
        return category_intervals

    @staticmethod
    def _parse_description(description: str) -> Tuple:
        birth_weight = [
            float(val) for val in description.split(", [")[1].split(")")[0].split(", ")
        ]
        gestational_age = [
            float(val) for val in description.split("- [")[1].split(")")[0].split(", ")
        ]
        return *birth_weight, *gestational_age
