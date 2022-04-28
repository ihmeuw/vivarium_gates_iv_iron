from typing import List, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium.framework.time import Time
from vivarium_public_health.risks import LBWSGDistribution
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    BIRTH_WEIGHT,
    GESTATIONAL_AGE,
)


class LBWSGExposure:

    randomness_stream_name = 'birth_propensities'
    birth_weight_pipeline_name = 'birth_weight.exposure'
    pregnancy_duration_pipeline_name = 'pregnancy_duration.exposure'
    exposure_parameters_pipeline_name = (
        "risk_factor.low_birth_weight_and_short_gestation.exposure_parameters"
    )

    configuration_defaults = {
        "low_birth_weight_and_short_gestation": {
            "exposure": "data",
            "rebinned_exposed": [],
            "category_thresholds": [],
        }
    }

    def __init__(self):
        self.lbwsg_exposure_distribution = LBWSGDistribution()
        self._sub_components = [self.lbwsg_exposure_distribution]

    def __repr__(self):
        return "LBWSGExposure()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "lbwsg_exposure"

    @property
    def sub_components(self) -> List:
        return self._sub_components

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.randomness_stream_name)
        self.birth_weight = self._get_birth_weight_pipeline(builder)
        self.pregnancy_duration = self._get_pregnancy_duration_pipeline(builder)

    def _get_birth_weight_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.birth_weight_pipeline_name,
            self._birth_weight_pipeline_source,
            requires_streams=[self.randomness_stream_name],
            requires_values=[self.exposure_parameters_pipeline_name],
        )

    def _get_pregnancy_duration_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            self.pregnancy_duration_pipeline_name,
            self._pregnancy_duration_pipeline_source,
            requires_streams=[self.randomness_stream_name],
            requires_values=[self.exposure_parameters_pipeline_name],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _birth_weight_pipeline_source(self, index: pd.Index) -> pd.Series:
        birth_weight_propensity = self.randomness.get_draw(index, additional_key='birth_weight')
        categorical_propensity = self.randomness.get_draw(index, additional_key='categorical')
        birth_weight = self.lbwsg_exposure_distribution.single_axis_ppf(
            BIRTH_WEIGHT, birth_weight_propensity, categorical_propensity
        )
        return birth_weight

    def _pregnancy_duration_pipeline_source(self, index: pd.Index) -> pd.Series:
        pregnancy_duration_propensity = self.randomness.get_draw(
            index, additional_key='pregnancy_duration'
        )
        categorical_propensity = self.randomness.get_draw(index, additional_key='categorical')
        pregnancy_duration = self.lbwsg_exposure_distribution.single_axis_ppf(
            GESTATIONAL_AGE, pregnancy_duration_propensity, categorical_propensity
        )
        return pregnancy_duration


def get_birth_date(
    conception_date: Union[Time, pd.Series],
    pregnancy_duration: Union[float, pd.Series]
) -> Union[Time, pd.Series]:
    return conception_date + pd.to_timedelta(pregnancy_duration, unit='d')
