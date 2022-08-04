import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_gates_iv_iron.constants import models, data_keys


class MaternalMortality:

    @property
    def name(self):
        return 'maternal_mortality'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)

        self.life_expectancy = builder.lookup.build_table(
            builder.data.load(data_keys.POPULATION.TMRLE),
            parameter_columns=['age'],
        )
        self.background_mortality_rate = builder.value.register_rate_producer(
            'background_mortality_rate',
            source=builder.lookup.build_table(
                self._get_background_mortality_rate(builder),
                key_columns=['sex'],
                parameter_columns=['age', 'year'],
            ),
        )

        self.probability_fatal_maternal_disorder = builder.lookup.build_table(
            builder.data.load(data_keys.MATERNAL_DISORDERS.PROBABILITY_FATAL),
            key_columns=['sex'],
            parameter_columns=['age', 'year'],
        )

        self.hemoglobin_maternal_disorders_risk_effect = builder.value.get_value("maternal_disorder_risk_effect")

        columns_required = [
            'alive',
            'exit_time',
            'pregnancy_status',
            'pregnancy_state_change_date',
            'pregnancy_duration',
        ]
        columns_created = ['cause_of_death', 'years_of_life_lost']
        self.population_view = builder.population.get_view(columns_created + columns_required)
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=columns_created
        )

        builder.event.register_listener(
            'time_step',
            self.on_time_step,
            priority=1,  # Yuck tight coupling.
        )

    def on_initialize_simulants(self, pop_data: SimulantData):
        pop_update = pd.DataFrame({
            'cause_of_death': 'not_dead',
            'years_of_life_lost': 0.,
        }, index=pop_data.index)
        self.population_view.update(pop_update)

    def on_time_step(self, event: Event):
        pop = self.population_view.get(event.index, query='alive == "alive"')

        pregnancy_ends_this_step = (
            (pop['pregnancy_status'] == models.PREGNANT_STATE)
            & (event.time - pop["pregnancy_state_change_date"] >= pop["pregnancy_duration"])
        )

        # Determine who dies
        draw = self.randomness.get_draw(pop.index, additional_key="maternal_disorder_death")
        p_fatal_maternal_disorder = (self.probability_fatal_maternal_disorder(pop.index)
                                     * self.hemoglobin_maternal_disorders_risk_effect(pop.index))
        p_fatal_maternal_disorder[p_fatal_maternal_disorder > 1.0] = 1.0
        would_die_due_to_maternal_disorders = draw < p_fatal_maternal_disorder
        died_due_to_maternal_disorders = (
            pregnancy_ends_this_step & would_die_due_to_maternal_disorders
        )
        died_due_to_background_causes_index = self.randomness.filter_for_rate(
            pop.index,
            rate=self.background_mortality_rate(pop.index),
            additional_key="other_cause_death",
        )
        died_due_to_background_causes = pd.Series(False, index=pop.index)
        died_due_to_background_causes.loc[died_due_to_background_causes_index] = True
        died_due_to_background_causes.loc[died_due_to_maternal_disorders] = False

        died_this_step = died_due_to_maternal_disorders | died_due_to_background_causes

        pop.loc[died_this_step, "alive"] = "dead"
        pop.loc[died_this_step, "exit_time"] = event.time
        pop.loc[died_this_step, "years_of_life_lost"] = (
            self.life_expectancy(pop.loc[died_this_step].index)
        )
        pop.loc[died_due_to_maternal_disorders, "cause_of_death"] = "maternal_disorders"
        pop.loc[died_due_to_background_causes, "cause_of_death"] = "other_causes"
        self.population_view.update(pop)

    @staticmethod
    def _get_background_mortality_rate(builder: Builder) -> pd.DataFrame:
        all_cause_mortality_data = builder.data.load(data_keys.POPULATION.ACMR)
        maternal_disorder_csmr = builder.data.load(data_keys.MATERNAL_DISORDERS.TOTAL_CSMR)
        idx = all_cause_mortality_data.columns.difference(['value']).tolist()
        background_mortality_rate = (
            all_cause_mortality_data.set_index(idx)
            - maternal_disorder_csmr.set_index(idx)
        ).reset_index()
        return background_mortality_rate
