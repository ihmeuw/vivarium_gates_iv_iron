import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import get_hash

from vivarium_gates_iv_iron.constants import data_values, models
from vivarium_gates_iv_iron.data import sampling


class MaternalBMIExposure:

    @property
    def name(self):
        return 'maternal_bmi_exposure'

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.hemoglobin = builder.value.get_value('hemoglobin.exposure')
        self.threshold = data_values.MATERNAL_BMI.ANEMIA_THRESHOLD
        p_low_anemic, p_low_non_anemic = self._sample_bmi_parameters(builder)
        self.probability_low_given_anemic = builder.lookup.build_table(
            p_low_anemic,
        )
        self.probability_low_given_non_anemic = builder.lookup.build_table(
            p_low_non_anemic,
        )

        self.population_view = builder.population.get_view([
            'pregnancy_status',
            'maternal_bmi_propensity',
        ])
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            requires_streams=[self.name],
            creates_columns=['maternal_bmi_propensity'],
        )

        self.maternal_bmi = builder.value.register_value_producer(
            'maternal_bmi',
            source=self.maternal_bmi,
            requires_columns=['pregnancy_status', 'maternal_bmi_propensity'],
        )

    def on_initialize_simulants(self, pop_data: SimulantData):
        propensity = self.randomness.get_draw(pop_data.index)
        propensity = propensity.rename('maternal_bmi_propensity')
        self.population_view.update(propensity)

    def maternal_bmi(self, index: pd.Index):
        pop = self.population_view.get(index)
        p = pop['maternal_bmi_propensity']
        p_low_anemic = self.probability_low_given_anemic(index)
        p_low_non_anemic = self.probability_low_given_non_anemic(index)

        pregnant = pop[pop['pregnancy_status'] == models.PREGNANT_STATE].index
        pregnant_hemoglobin = self.hemoglobin(pregnant)
        anemic = pregnant[pregnant_hemoglobin < self.threshold]
        non_anemic = pregnant.difference(anemic)

        bmi = pd.Series(models.INVALID_BMI, index=index, name='maternal_bmi')
        bmi[anemic] = np.where(
            p.loc[anemic] < p_low_anemic.loc[anemic], models.LOW_BMI, models.NORMAL_BMI,
        )
        bmi[non_anemic] = np.where(
            p.loc[non_anemic] < p_low_non_anemic.loc[non_anemic], models.LOW_BMI, models.NORMAL_BMI,
        )
        return bmi

    ###########
    # Helpers #
    ###########

    def _sample_bmi_parameters(self, builder: Builder):
        draw = builder.configuration.input_data.input_draw_number
        location = builder.configuration.input_data.location
        seed = f'bmi_{draw}_{location}'
        np.random.seed(get_hash(seed))
        probabilities = []
        for params in [data_values.MATERNAL_BMI.PROBABILITY_LOW_BMI_ANEMIC,
                       data_values.MATERNAL_BMI.PROBABILITY_LOW_BMI_NON_ANEMIC]:
            dist = sampling.get_lognorm_from_quantiles(*params)
            probabilities.append(dist.rvs())
        return tuple(probabilities)
