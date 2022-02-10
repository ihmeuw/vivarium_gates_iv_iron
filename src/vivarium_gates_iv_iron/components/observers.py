import itertools
from typing import Callable, Dict, Iterable, List, Tuple, Union

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_public_health.metrics import (MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_,
                                            DiseaseObserver as DiseaseObserver_,
                                            CategoricalRiskObserver as CategoricalRiskObserver_)
from vivarium_public_health.metrics.utilities import (get_deaths, get_state_person_time, get_transition_count,
                                                      get_years_lived_with_disability, get_years_of_life_lost,
                                                      get_person_time,
                                                      TransitionString)

from vivarium_gates_iv_iron.constants import models, results, data_keys


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def __init__(self, observer_name: str = 'False'):
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
                columns_required.append(data_keys.WASTING.name)

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

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder') -> None:

        #TODO: pregnancy state transition counts
        #TODO: pregnancy state person time
        #TODO: pregnancy outcomes (count: live birht, stillbirth, other)

        #TODO: add new stratification

        config = builder.configuration.metrics
        self.observation_start = pd.Timestamp(**config.observation_start)

        builder.event.register_listener('collect_metrics', self.on_collect_metrics)
        builder.event.register_listener('simulation_end', self.on_simulation_end)

        builder.value.register_value_modifier('metrics', self.metrics)

    def on_collect_metrics(self, event: 'Event') -> None:
        pop = self.get_denominator_pop(event)
        states = list(models.MULTIPLE_MYELOMA_WITH_CONDITION_STATES)

        for current_state, next_state in zip(states, states[1:] + [states[-1]]):
            state_denominator = self.subset_state_denominator(pop, current_state, next_state, event)
            if not state_denominator.empty:
                print(f'current state: {current_state}, state denominator size: {len(state_denominator)}')
            for risk_status in itertools.product(*data_values.RISK_LEVEL_MAP.values()):
                denominator = self.subset_risk_denominator(state_denominator, risk_status)

                alive_at_start = (denominator
                                  .groupby('group')
                                  .multiple_myeloma
                                  .count()
                                  .rename('alive'))
                died_by_end = (denominator[denominator['exit_time'] == event.time]
                               .groupby('group')
                               .multiple_myeloma
                               .count()
                               .rename('died'))
                progressed_by_end = (denominator[denominator[f'{next_state}_event_time'] == event.time]
                                     .groupby('group')
                                     .multiple_myeloma
                                     .count()
                                     .rename('progressed'))
                survival_results = pd.concat([alive_at_start, died_by_end, progressed_by_end], axis=1)
                survival_results.index = pd.IntervalIndex(survival_results.index)
                if not denominator.empty:
                    print(f'risk_status: {risk_status}, denominator size: {len(denominator)}, summary: {survival_results.sum().to_dict()}')
                treatment_line = current_state.split('_')[-1]
                for interval, interval_data in survival_results.iterrows():
                    for measure, template in self.templates:
                        key = template.format(
                            treatment_line=treatment_line,
                            period_start=interval.left,
                            period_end=interval.right,
                            **dict(zip(data_values.RISKS, risk_status))
                        )
                        self.counts[key] += interval_data.loc[measure]

    def on_simulation_end(self, event: 'Event'):
        pop = self.get_denominator_pop(event)
        states = list(models.MULTIPLE_MYELOMA_WITH_CONDITION_STATES)

        for current_state, next_state in zip(states, states[1:] + [states[-1]]):
            for risk_status in itertools.product(*data_values.RISK_LEVEL_MAP.values()):
                denominator = self.subset_risk_denominator(pop, risk_status)
                denominator = self.subset_state_denominator(denominator, current_state, next_state, event)
                right_censored_mask = ~(
                    (denominator['exit_time'] != event.time)
                    | (denominator[f'{next_state}_event_time'] == event.time)
                )
                right_censored = (denominator[right_censored_mask]
                                  .groupby('group')
                                  .multiple_myeloma
                                  .count()
                                  .rename('right_censored'))
                treatment_line = current_state.split('_')[-1]
                for interval, count in right_censored.iteritems():
                    key = self.sim_end_template.format(
                        treatment_line=treatment_line,
                        period_end=interval.right,
                        **dict(zip(data_values.RISKS, risk_status))
                    )
                    self.counts[key] += count

    def metrics(self, index: pd.Index, metrics: Dict[str, float]) -> Dict[str, float]:
        metrics.update(self.counts)
        return metrics

    def get_denominator_pop(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        living = pop['alive'] == 'alive'
        died_this_step = (pop['alive'] == 'dead') & (pop['exit_time'] == event.time)
        in_denominator = living | died_this_step
        return pop.loc[in_denominator]

    def subset_risk_denominator(self, pop: pd.DataFrame, risk_status):
        risk_mask = pd.Series(True, index=pop.index)
        for risk, risk_level in zip(data_values.RISKS, risk_status):
            risk_mask &= pop[risk] == risk_level
        return pop.loc[risk_mask]

    def subset_state_denominator(self, pop: pd.DataFrame, current_state: str, next_state: str, event: 'Event'):
        left_censored = (pop[f'{current_state}_event_time'].notnull()
                         & (pop[f'{current_state}_event_time'] < self.observation_start))
        in_current_state_denominator = ~left_censored & (
            # In the current state and didn't get there this time step
            ((pop[models.MULTIPLE_MYELOMA_MODEL_NAME] == current_state)
             & (pop[f'{current_state}_event_time'] < event.time))
            # or in the next state, but not til the start of the next step.
            | ((pop[models.MULTIPLE_MYELOMA_MODEL_NAME] == next_state)
               & (pop[f'{next_state}_event_time'] == event.time))
        )

        denominator = pop.loc[in_current_state_denominator].copy()
        denominator['group'] = pd.cut(denominator[f'{current_state}_time_since_entrance'], self.bins)
        return denominator
