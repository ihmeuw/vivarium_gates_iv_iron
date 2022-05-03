from pathlib import Path
from typing import List, Dict

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.values import Pipeline
from vivarium_public_health.risks import LBWSGDistribution
from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    BIRTH_WEIGHT,
    GESTATIONAL_AGE,
)

from vivarium_gates_iv_iron import paths
from vivarium_gates_iv_iron.constants import models


class LBWSGExposure:

    randomness_stream_name = 'birth_propensities'
    sex_of_child_column_name = 'sex_of_child'
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
        self.pregnancy_duration_in_weeks = self._get_pregnancy_duration_pipeline(builder)
        self.population_view = self.get_population_view(builder)

        self._register_simulant_initializer(builder)

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

    def get_population_view(self, builder: Builder) -> PopulationView:
        return builder.population.get_view([self.sex_of_child_column_name])

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=[self.sex_of_child_column_name],
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(
            pd.Series(
                models.INVALID_OUTCOME, index=pop_data.index, name=self.sex_of_child_column_name
            )
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


class BirthObserver:

    pregnancy_status_column_name = "pregnancy_status"
    pregnancy_duration_column_name = "pregnancy_duration"
    pregnancy_state_change_column_name = "pregnancy_state_change_date"
    child_sex_column_name = "sex_of_child"

    birth_date_column_name = "birth_date"
    sex_column_name = "sex"
    birth_weight_column_name = "birth_weight"
    gestational_age_column_name = "gestational_age"

    birth_weight_pipeline_name = 'birth_weight.exposure'

    def __repr__(self):
        return "BirthObserver()"

    ##############
    # Properties #
    ##############

    @property
    def name(self):
        return "birth_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.location = Path(builder.configuration.input_data.artifact_path).stem
        self.input_draw = builder.configuration.input_data.input_draw_number
        self.seed = builder.configuration.randomness.random_seed
        self.scenario = builder.configuration.intervention.scenario

        self.pipelines = self.get_pipelines(builder)
        self.population_view = self.get_population_view(builder)

        # todo add other attributes to be tracked as needed to both dataframes
        self.ongoing_pregnancies = pd.DataFrame(
            {
                self.birth_date_column_name: [],
                self.sex_column_name: [],
                self.birth_weight_column_name: [],
                self.gestational_age_column_name: [],
            }
        )
        self.births = self.ongoing_pregnancies.copy()

        self._register_simulant_initializer(builder)
        self._register_collect_metrics_listener(builder)
        self._register_simulation_end_listener(builder)

    def get_pipelines(self, builder: Builder) -> Dict[str, Pipeline]:
        return {
            self.birth_weight_pipeline_name:
                builder.value.get_value(self.birth_weight_pipeline_name),
        }

    def get_population_view(self, builder: Builder) -> PopulationView:
        columns_required = [
            self.pregnancy_status_column_name,
            self.pregnancy_duration_column_name,
            self.pregnancy_state_change_column_name,
            self.child_sex_column_name,
        ]
        return builder.population.get_view(columns_required)

    def _register_simulant_initializer(self, builder: Builder) -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_columns=[
                self.child_sex_column_name,
                self.pregnancy_status_column_name,
                self.pregnancy_duration_column_name,
                self.pregnancy_state_change_column_name,
            ],
            requires_values=[self.birth_weight_pipeline_name]
        )

    def _register_collect_metrics_listener(self, builder: Builder) -> None:
        builder.event.register_listener("time_step", self.on_collect_metrics)

    def _register_simulation_end_listener(self, builder: Builder) -> None:
        builder.event.register_listener("simulation_end", self.write_output)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self._record_conceptions(self.population_view.get(pop_data.index))

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        self._record_births(pop, event)
        self._record_conceptions(pop, event)

    # noinspection PyUnusedLocal
    def write_output(self, event: Event) -> None:
        filename = f"{self.location}_{self.scenario}_{self.input_draw}_{self.seed}.hdf"
        output_path = paths.CHILD_DATA_OUTPUT_DIR / filename
        self.births.to_hdf(output_path, "child_birth_data")

    ##################
    # Helper methods #
    ##################

    def _record_births(self, pop: pd.DataFrame, event: Event) -> None:
        new_births = self.ongoing_pregnancies[
            self.ongoing_pregnancies[self.birth_date_column_name] <= event.time
        ]

        # this intersection removes all children whose mothers have passed away during pregnancy
        live_births = new_births[new_births.index.intersection(pop.index)]

        self.births = pd.concat([self.births, live_births], ignore_index=True)
        self.ongoing_pregnancies = self.ongoing_pregnancies.drop(new_births.index)

    def _record_conceptions(self, pop: pd.DataFrame, event: Event = None) -> None:
        conception_mask = pop[self.pregnancy_status_column_name] == models.PREGNANT_STATE
        if event is not None:
            conception_mask &= (pop[self.pregnancy_state_change_column_name == event.time])

        conception_index = pop[conception_mask].index
        pregnancy_duration = pop.loc[conception_index, self.pregnancy_duration_column_name]

        child_sex = (
            pop.loc[conception_index, self.child_sex_column_name]
            .rename(self.sex_column_name)
        )
        birth_date = (
            pop.loc[conception_index, self.pregnancy_state_change_column_name] + pregnancy_duration
        )
        birth_weight = self.pipelines[self.birth_weight_pipeline_name](conception_index)
        gestational_age = (
            pregnancy_duration
            .apply(lambda td: (td.days + td.seconds / (3600 * 24)) / 7.0)
            .rename(self.gestational_age_column_name)
        )

        new_conceptions = pd.concat([child_sex, birth_date, birth_weight, gestational_age], axis=1)
        self.ongoing_pregnancies = pd.concat([self.ongoing_pregnancies, new_conceptions])
