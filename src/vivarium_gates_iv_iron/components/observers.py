from collections import Counter
import itertools
from typing import Callable, Dict, Iterable, List, Tuple, Union

import pandas as pd
from vivarium import ConfigTree
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import PopulationView
from vivarium.framework.time import Time, get_time_stamp
from vivarium.framework.values import Pipeline
from vivarium_public_health.metrics import (utilities,
                                            MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_,
                                            DiseaseObserver as DiseaseObserver_,
                                            CategoricalRiskObserver as CategoricalRiskObserver_)
from vivarium_public_health.metrics.utilities import (get_group_counts, get_output_template, get_deaths, get_state_person_time, get_transition_count,
                                                      get_years_lived_with_disability, get_years_of_life_lost,
                                                      get_person_time,
                                                      QueryString, TransitionString)

from vivarium_gates_iv_iron.constants import models, results, data_keys


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def __init__(self, observer_name: str = 'False', by_pregnancy_outcome: str = 'False'):
        self.name = f'{observer_name}_results_stratifier'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Perform this component's setup."""
        # The only thing you should request here are resources necessary for results stratification.
        self.pipelines = {}
        columns_required = ['tracked']

        self.stratification_levels = {}

        def setup_stratification(source_name: str, is_pipeline: bool, stratification_name: str,
                                 categories: Iterable):

            def get_state_function(state: Union[str, bool, List]) -> Callable:
                return lambda pop: (pop[source_name] == state if not isinstance(state, List)
                                    else pop[source_name].isin(state))

            if type(categories) != dict:
                categories = {category: category for category in categories}

            self.stratification_levels[stratification_name] = {
                stratification_key: get_state_function(source_value)
                for stratification_key, source_value in categories.items()
            }

            if is_pipeline:
                self.pipelines[source_name] = builder.value.get_value(source_name)
            else:
                columns_required.append(source_name)

        self.population_view = builder.population.get_view(columns_required)
        self.stratification_groups: pd.Series = None

        # Ensure that the stratifier updates before its observer
        builder.event.register_listener('time_step__prepare', self.on_timestep_prepare, priority=0)

    # noinspection PyAttributeOutsideInit
    def on_timestep_prepare(self, event: Event):
        # cache stratification groups at the beginning of the time-step for use later when stratifying
        self.stratification_groups = self.get_stratification_groups(event.index)

    def get_stratification_groups(self, index: pd.Index):
        #  get values required for stratification from population view and pipelines
        pop_list = [self.population_view.get(index)] + [pd.Series(pipeline(index), name=name)
                                                        for name, pipeline in self.pipelines.items()]
        pop = pd.concat(pop_list, axis=1)

        stratification_groups = pd.Series('', index=index)
        all_stratifications = self.get_all_stratifications()
        for stratification in all_stratifications:
            stratification_group_name = '_'.join([f'{metric["metric"]}_{metric["category"]}'
                                                  for metric in stratification]).lower()
            mask = pd.Series(True, index=index)
            for metric in stratification:
                mask &= self.stratification_levels[metric['metric']][metric['category']](pop)
            stratification_groups.loc[mask] = stratification_group_name
        return stratification_groups

    def get_all_stratifications(self) -> List[Tuple[Dict[str, str], ...]]:
        """
        Gets all stratification combinations. Returns a List of Stratifications. Each Stratification is represented as a
        Tuple of Stratification Levels. Each Stratification Level is represented as a Dictionary with keys 'metric' and
        'category'. 'metric' refers to the stratification level's name, and 'category' refers to the stratification
        category.

        If no stratification levels are defined, returns a List with a single empty Tuple
        """
        # Get list of lists of metric and category pairs for each metric
        groups = [[{'metric': metric, 'category': category} for category in category_maps]
                  for metric, category_maps in self.stratification_levels.items()]
        # Get product of all stratification combinations
        return list(itertools.product(*groups))

    @staticmethod
    def get_stratification_key(stratification: Iterable[Dict[str, str]]) -> str:
        return ('' if not stratification
                else '_'.join([f'{metric["metric"]}_{metric["category"]}' for metric in stratification]))

    def group(self, pop: pd.DataFrame) -> Iterable[Tuple[Tuple[str, ...], pd.DataFrame]]:
        """Takes the full population and yields stratified subgroups.

        Parameters
        ----------
        pop
            The population to stratify.

        Yields
        ------
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        index = pop.index.intersection(self.stratification_groups.index)
        pop = pop.loc[index]
        stratification_groups = self.stratification_groups.loc[index]

        stratifications = self.get_all_stratifications()
        for stratification in stratifications:
            stratification_key = self.get_stratification_key(stratification)
            if pop.empty:
                pop_in_group = pop
            else:
                pop_in_group = pop.loc[(stratification_groups == stratification_key)]
            yield (stratification_key,), pop_in_group

    @staticmethod
    def update_labels(measure_data: Dict[str, float], labels: Tuple[str, ...]) -> Dict[str, float]:
        """Updates a dict of measure data with stratification labels.

        Parameters
        ----------
        measure_data
            The measure data with unstratified column names.
        labels
            The stratification labels. Yielded along with the population
            subgroup the measure data was produced from by a call to
            :obj:`ResultsStratifier.group`.

        Returns
        -------
            The measure data with column names updated with the stratification
            labels.

        """
        stratification_label = f'_{labels[0]}' if labels[0] else ''
        measure_data = {f'{k}{stratification_label}': v for k, v in measure_data.items()}
        return measure_data


class MortalityObserver(MortalityObserver_):

    def __init__(self):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def metrics(self, index: pd.Index, metrics: Dict[str, float]) -> Dict[str, float]:
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        measure_getters = (
            (get_deaths, (self.causes,)),
            (get_person_time, ()),
            (get_years_of_life_lost, (self.life_expectancy, self.causes)),
        )

        for labels, pop_in_group in self.stratifier.group(pop):
            base_args = (pop_in_group, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)

            for measure_getter, extra_args in measure_getters:
                measure_data = measure_getter(*base_args, *extra_args)
                measure_data = self.stratifier.update_labels(measure_data, labels)
                metrics.update(measure_data)

        # TODO remove stratification by wasting state of deaths/ylls due to PEM?

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics[results.TOTAL_YLLS_COLUMN] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics


class DisabilityObserver(DisabilityObserver_):

    def __init__(self):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        self.update_metrics(pop)

        pop.loc[:, results.TOTAL_YLDS_COLUMN] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    def update_metrics(self, pop: pd.DataFrame):
        for labels, pop_in_group in self.stratifier.group(pop):
            base_args = (pop_in_group, self.config.to_dict(), self.clock().year, self.step_size(), self.age_bins,
                         self.disability_weight_pipelines, self.causes)
            measure_data = self.stratifier.update_labels(get_years_lived_with_disability(*base_args), labels)
            self.years_lived_with_disability.update(measure_data)


class PregnancyObserver:
    configuration_defaults = {
        'metrics': {
            'pregnancy': {
                'by_age': True,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __repr__(self):
        return 'PregnancyObserver()'


    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> List:
        return []

    @property
    def name(self):
        return 'pregnancy_observer'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.configuration = builder.configuration.metrics.pregnancy
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.age_bins = utilities.get_age_bins(builder)
        self.person_time = Counter()
        self.counts = Counter()

        columns_required = ['alive', 'pregnancy_status', 'pregnancy_outcome', 'pregnancy_state_change_date']
        if self.configuration.by_age:
            columns_required += ['age']
        if self.configuration.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)
        builder.value.register_value_modifier('metrics', self.metrics)


    def on_time_step_prepare(self, event: Event):
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for state in models.PREGNANCY_MODEL_STATES:
            # noinspection PyTypeChecker
            state_person_time_this_step = utilities.get_state_person_time(
                pop, self.configuration, 'pregnancy_status', state, self.clock().year, event.step_size, self.age_bins
            )
            self.person_time.update(state_person_time_this_step)

    def on_collect_metrics(self, event: Event):
        counts_this_step = {}
        pop = self.population_view.get(event.index)
        pop = pop[pop["pregnancy_state_change_date"] == event.time]
        configuration = self.configuration.to_dict()

        # count new conceptions
        base_key = get_output_template(**configuration).substitute(measure='conception_count',
                                                            year=event.time.year)
        base_filter = QueryString(f'alive == "alive" and pregnancy_status == "pregnant"')
        counts_this_step.update(get_group_counts(pop, base_filter, base_key, self.configuration, self.age_bins))
        for outcome in models.PREGNANCY_OUTCOMES:
            base_key = get_output_template(**configuration).substitute(measure=f'{outcome}_count',
                                                                          year=event.time.year)
            base_filter = QueryString(
                f'alive == "alive" and pregnancy_status == "postpartum" and pregnancy_outcome == "{outcome}"')
            counts_this_step.update(get_group_counts(pop, base_filter, base_key, self.configuration, self.age_bins))

        self.counts.update(counts_this_step)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    # noinspection PyUnusedLocal
    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics


class MaternalDisordersObserver:
    configuration_defaults = {
        'metrics': {
            'maternal_disorders': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __repr__(self):
        return 'MaternalDisordersObserver()'


    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> List:
        return []

    @property
    def name(self):
        return 'maternal_disorders_observer'

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.configuration = builder.configuration.metrics.maternal_disorders
        self.start_time = get_time_stamp(builder.configuration.time.start)
        self.age_bins = utilities.get_age_bins(builder)
        self.deaths = Counter()
        self.counts = Counter()

        columns_required = ['alive', 'exit_time', 'cause_of_death', 'pregnancy_status', 'pregnancy_state_change_date']
        if self.configuration.by_age:
            columns_required += ['age']
        if self.configuration.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.event.register_listener('collect_metrics', self.on_collect_metrics)
        builder.value.register_value_modifier('metrics', self.metrics)


    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index)
        configuration = self.configuration.to_dict()

        # count deaths due to maternal disorders
        deaths_this_step = {}
        died_this_step_pop = pop[pop["exit_time"] == event.time]
        death_key = get_output_template(**configuration).substitute(measure='death_due_to_maternal_disorders',
                                                            year=event.time.year)
        death_filter = QueryString(f'alive == "dead" and cause_of_death == "maternal_disorders"')
        deaths_this_step.update(get_group_counts(died_this_step_pop, death_filter, death_key, self.configuration, self.age_bins))
        self.deaths.update(deaths_this_step)

        # count incident cases of to maternal disorders
        cases_this_step = {}
        pregnancy_change_this_step_pop = pop[pop["pregnancy_state_change_date"] == event.time]
        case_key = get_output_template(**configuration).substitute(measure='incident_cases_of_maternal_disorders',
                                                            year=event.time.year)
        case_filter = QueryString(f'(alive=="alive" and pregnancy_status == "maternal_disorder") or '
                                  f'(alive == "dead" and cause_of_death == "maternal_disorders")')
        cases_this_step.update(get_group_counts(pregnancy_change_this_step_pop, case_filter, case_key, self.configuration, self.age_bins))
        self.counts.update(cases_this_step)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    # noinspection PyUnusedLocal
    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        metrics.update(self.deaths)
        metrics.update(self.counts)
        return metrics