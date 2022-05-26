from collections import Counter
import itertools
from typing import Dict, List, Iterable, Tuple

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health.metrics import (
    DisabilityObserver as DisabilityObserver_,
    MortalityObserver as MortalityObserver_,
    ResultsStratifier as ResultsStratifier_,
)
from vivarium_public_health.metrics.stratification import Source, SourceType, StratificationLevel
from vivarium_public_health.utilities import to_years

from vivarium_gates_iv_iron.constants import data_values, models


class ResultsStratifier(ResultsStratifier_):

    def register_stratifications(self, builder: Builder) -> None:
        start_year = builder.configuration.time.start.year
        end_year = builder.configuration.time.end.year

        self.setup_stratification(
            builder,
            name=ResultsStratifier.YEAR,
            sources=[ResultsStratifier.YEAR_SOURCE],
            categories={str(year) for year in range(start_year, end_year + 1)},
            mapper=self.year_stratification_mapper,
            current_category_getter=self.year_current_categories_getter,
        )

        # self.setup_stratification(
        #     builder,
        #     name=ResultsStratifier.SEX,
        #     sources=[ResultsStratifier.SEX_SOURCE],
        #     categories=ResultsStratifier.SEX_CATEGORIES,
        # )

        # self.setup_stratification(
        #     builder,
        #     name=ResultsStratifier.AGE,
        #     sources=[ResultsStratifier.AGE_SOURCE],
        #     categories={age_bin for age_bin in self.age_bins["age_group_name"]},
        #     mapper=self.age_stratification_mapper,
        # )

        self.setup_stratification(
            builder,
            name='pregnancy_status',
            sources=[Source('pregnancy_status', SourceType.COLUMN)],
            categories=models.PREGNANCY_MODEL_STATES,
        )

        self.setup_stratification(
            builder,
            name='maternal_supplementation',
            sources=[Source('maternal_supplementation', SourceType.COLUMN)],
            categories=models.SUPPLEMENTATION_CATEGORIES,
        )

        self.setup_stratification(
            builder,
            name='antenatal_iv_iron',
            sources=[Source('antenatal_iv_iron', SourceType.COLUMN)],
            categories=models.IV_IRON_TREATMENT_STATUSES,
        )

        self.setup_stratification(
            builder,
            name='postpartum_iv_iron',
            sources=[Source('postpartum_iv_iron', SourceType.COLUMN)],
            categories=models.IV_IRON_TREATMENT_STATUSES,
        )

    def _get_current_stratifications(
        self, include: Iterable[str], exclude: Iterable[str]
    ) -> List[Tuple[Tuple[StratificationLevel, str], ...]]:
        """
        Gets all stratification combinations. Returns a List of
        Stratifications. Each Stratification is represented as a Tuple of
        Levels. Each Level is represented as a Tuple of a StratificationLevel
        object and string referring to the specific stratification category.

        If no stratification levels are defined, returns a List with a single empty Tuple
        """
        include = set(include)
        exclude = set(exclude)
        level_names = (self.default_stratification_levels | include) - exclude

        groups = []
        for level_name in level_names:
            try:
                level = self.stratification_levels[level_name]
            except KeyError:
                raise KeyError(
                    f"Stratification by {level_name} is not defined. "
                    f"Available stratifications are {list(self.stratification_levels)}.")
            groups.append(
                [(level, category) for category in level.current_categories]
            )

        # Get product of all stratification combinations
        return list(itertools.product(*groups))


class MortalityObserver(MortalityObserver_):

    def setup(self, builder: Builder):
        super().setup(builder)
        self.causes_of_death += ['maternal_disorders']


class DisabilityObserver(DisabilityObserver_):

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.disability_pipelines['maternal_disorders'] = builder.value.get_value(
            'maternal_disorders.disability_weight'
        )
        self.disability_pipelines['anemia'] = builder.value.get_value(
            'real_anemia.disability_weight'
        )

    def on_time_step_prepare(self, event: Event) -> None:
        step_size_in_years = to_years(event.step_size)
        pop = self.population_view.get(
            event.index, query='tracked == True and alive == "alive"'
        )

        new_observations = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for cause, disability_weight in self.disability_pipelines.items():
            cause_dw = disability_weight(pop.index)
            for label, group_mask in groups:
                key = f"ylds_due_to_{cause}_{label}"
                new_observations[key] = cause_dw[group_mask].sum() * step_size_in_years
        self.counts.update(new_observations)

        pop.loc[:, self.ylds_column_name] += self.disability_weight(pop.index)
        self.population_view.update(pop)


class PregnancyObserver:

    configuration_defaults = {
        "observers": {
            "pregnancy": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "PregnancyObserver()"

    @property
    def name(self):
        return "pregnancy_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.observers.pregnancy
        self.stratifier = builder.components.get_component(ResultsStratifier.name)

        self.person_time = Counter()
        self.counts = Counter()

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=['previous_pregnancy_status'],
            requires_columns=['pregnancy_status']
        )

        columns_required = [
            "alive",
            "exit_time",
            "cause_of_death",
            "pregnancy_status",
            "pregnancy_outcome",
            "pregnancy_state_change_date",
            "maternal_hemorrhage",
            "previous_pregnancy_status",
        ]

        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.event.register_listener("collect_metrics", self.on_collect_metrics)
        builder.value.register_value_modifier("metrics", self.metrics)

    def on_initialize_simulants(self, pop_data: SimulantData):
        pop = self.population_view.subview(['pregnancy_status']).get(pop_data.index)
        self.population_view.update(
            pop.rename(columns={'pregnancy_status': 'previous_pregnancy_status'})
        )

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        step_size = to_years(event.step_size)

        pregnancy_measures = list(itertools.product(
            models.PREGNANCY_MODEL_STATES,
            models.PREGNANCY_OUTCOMES,
        ))

        new_person_time = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            group = pop[group_mask]
            for hemorrhage_type in models.MATERNAL_HEMORRHAGE_STATES:
                for state, outcome in pregnancy_measures:
                    key = f"{state}_with_{outcome}_with_{hemorrhage_type}_person_time_{label}"
                    sub_group = group.query(
                        f'pregnancy_status == "{state}" '
                        f'and pregnancy_outcome == "{outcome}" '
                        f'and maternal_hemorrhage == "{hemorrhage_type}"'
                    )
                    new_person_time[key] = len(sub_group) * step_size

        self.person_time.update(new_person_time)

        # This enables tracking of transitions between states
        pop['previous_pregnancy_status'] = pop["pregnancy_status"]
        self.population_view.update(pop)

    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index)
        # Might have some dead pops here, but they'll have died this time step.
        pop = pop[pop["pregnancy_state_change_date"] == event.time]

        counts_this_step = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            group = pop[group_mask]
            for transition in models.PREGNANCY_MODEL_TRANSITIONS:
                key = f'{transition}_count_{label}'
                sub_group = group.query(
                    f'previous_pregnancy_status == "{transition.from_state}" '
                    f'and pregnancy_status == "{transition.to_state}"'
                )
                counts_this_step[key] = len(sub_group)

            for outcome in models.PREGNANCY_OUTCOMES:
                key = f'{outcome}_count_{label}'
                sub_group = group.query(
                    f'pregnancy_outcome == "{outcome}" '
                    f'and (pregnancy_status == "{models.MATERNAL_DISORDER_STATE}" '
                    f'or pregnancy_status == "{models.NO_MATERNAL_DISORDER_STATE}")'
                )
                counts_this_step[key] = len(sub_group)

            key = f'incident_cases_of_maternal_disorders_{label}'
            sub_group = group.query(
                f'pregnancy_status == "{models.MATERNAL_DISORDER_STATE}"'
            )
            counts_this_step[key] = len(sub_group)

            for hemorrhage_status in models.MATERNAL_HEMORRHAGE_STATES[:-1]:
                key = f"incident_cases_of_{hemorrhage_status}_{label}"
                sub_group = group.query(
                    f'pregnancy_status != "{models.POSTPARTUM_STATE}" '
                    f'and maternal_hemorrhage == "{hemorrhage_status}"'
                )
                counts_this_step[key] = len(sub_group)

        self.counts.update(counts_this_step)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    # noinspection PyUnusedLocal
    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics


class AnemiaObserver:

    configuration_defaults = {
        "observers": {
            "anemia": {
                "exclude": [],
                "include": [],
            }
        }
    }

    def __repr__(self):
        return "AnemiaObserver()"

    @property
    def name(self):
        return "anemia_observer"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.observers.anemia
        self.stratifier = builder.components.get_component(ResultsStratifier.name)

        self.person_time = Counter()
        self.exposure = Counter()

        self.anemia_levels = builder.value.get_value("anemia_levels")
        self.hemoglobin = builder.value.get_value("hemoglobin.exposure")

        columns_required = [
            "alive",
            "pregnancy_status",
            "maternal_hemorrhage",
        ]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.event.register_listener("collect_metrics", self.on_collect_metrics)
        builder.value.register_value_modifier("metrics", self.metrics)

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        pop["anemia_level"] = self.anemia_levels(event.index)
        step_size = to_years(event.step_size)

        anemia_measures = list(itertools.product(
            data_values.ANEMIA_DISABILITY_WEIGHTS.keys(),
            models.PREGNANCY_MODEL_STATES,
            models.MATERNAL_HEMORRHAGE_STATES,
        ))

        new_person_time = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            group = pop[group_mask]
            for anemia_level, pregnancy_status, hemorrhage_state in anemia_measures:
                key = f"{anemia_level}_anemia_person_time_among_{pregnancy_status}_with_{hemorrhage_state}_{label}"
                sub_group = group.query(
                    f'anemia_level == "{anemia_level}" '
                    f'and pregnancy_status == "{pregnancy_status}" '
                    f'and maternal_hemorrhage == "{hemorrhage_state}"'
                )
                new_person_time[key] = len(sub_group) * step_size

        self.person_time.update(new_person_time)

    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        pop["hemoglobin"] = self.hemoglobin(event.index)

        pregnancy_measures = list(itertools.product(
            models.PREGNANCY_MODEL_STATES,
            models.MATERNAL_HEMORRHAGE_STATES,
        ))

        new_exposures = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            group = pop[group_mask]
            for pregnancy_status, hemorrhage_state in pregnancy_measures:
                key = f"hemoglobin_exposure_sum_among_{pregnancy_status}_with_{hemorrhage_state}_{label}"
                sub_group = group.query(
                    f'pregnancy_status == "{pregnancy_status}" '
                    f'and maternal_hemorrhage == "{hemorrhage_state}"'
                )
                new_exposures[key] = sub_group.hemoglobin.sum()

        self.exposure.update(new_exposures)

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.person_time)
        metrics.update(self.exposure)
        return metrics


class MaternalBMIObserver:

    configuration_defaults = {
        "observers": {
            "maternal_bmi": {
                "exclude": [],
                "include": [],
            }
        }
    }

    @property
    def name(self):
        return "maternal_bmi_observer"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.observers.maternal_bmi
        self.stratifier = builder.components.get_component(ResultsStratifier.name)
        self.bmi_person_time = Counter()

        columns_required = [
            "alive",
            "pregnancy_status",
            "maternal_bmi_anemia_category",
        ]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.value.register_value_modifier("metrics", self.metrics)

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        step_size = to_years(event.step_size)

        new_person_time = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)
        for label, group_mask in groups:
            group = pop[group_mask]
            for bmi_cat in models.BMI_ANEMIA_CATEGORIES:
                key = f"bmi_person_time_{bmi_cat}_{label}"
                sub_group = group.query(f'maternal_bmi_anemia_category == "{bmi_cat}"')
                new_person_time[key] = len(sub_group) * step_size

        self.bmi_person_time.update(new_person_time)

    def metrics(self, index: pd.Index, metrics: Dict):
        metrics.update(self.bmi_person_time)
        return metrics


class InterventionObserver:

    configuration_defaults = {
        "observers": {
            "intervention": {
                "exclude": [],
                "include": [],
            }
        }
    }

    @property
    def name(self):
        return "intervention_observer"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.observers.intervention
        self.stratifier = builder.components.get_component(ResultsStratifier.name)
        self.person_time = Counter()

        columns_required = [
            "alive",
            "pregnancy_status",
            "maternal_bmi_anemia_category",
            "maternal_supplementation",
            "antenatal_iv_iron",
            "postpartum_iv_iron",
        ]
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.value.register_value_modifier("metrics", self.metrics)

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        step_size = to_years(event.step_size)

        new_person_time = {}
        groups = self.stratifier.group(pop.index, self.config.include, self.config.exclude)

        for label, group_mask in groups:
            group = pop[group_mask]
            for treatment in ['antenatal_iv_iron', 'postpartum_iv_iron']:
                for treatment_status in models.IV_IRON_TREATMENT_STATUSES:
                    count = 0
                    for bmi_cat in models.BMI_ANEMIA_CATEGORIES:
                        key = f"person_time_{treatment}_{treatment_status}_bmi_{bmi_cat}_{label}"
                        sub_group = group.query(f'{treatment} == "{treatment_status}" '
                                                f'and maternal_bmi_anemia_category == "{bmi_cat}"')
                        new_person_time[key] = len(sub_group) * step_size
                        count += len(sub_group)

            for treatment_status in models.SUPPLEMENTATION_CATEGORIES:
                for bmi_cat in models.BMI_ANEMIA_CATEGORIES:
                    key = f"person_time_maternal_supplementation_{treatment_status}_bmi_{bmi_cat}_{label}"
                    sub_group = group.query(
                        f'maternal_supplementation == "{treatment_status}"'
                        f'and maternal_bmi_anemia_category == "{bmi_cat}"'
                    )
                    new_person_time[key] = len(sub_group) * step_size

        self.person_time.update(new_person_time)

    def metrics(self, index: pd.Index, metrics: Dict):
        metrics.update(self.person_time)
        return metrics
