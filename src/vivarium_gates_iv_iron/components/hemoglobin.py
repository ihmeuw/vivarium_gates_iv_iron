import numpy as np
import pandas as pd
import scipy.stats

from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


from vivarium_gates_iv_iron.constants.data_values import HEMOGLOBIN_DISTRIBUTION_PARAMETERS, HEMOGLOBIN_THRESHOLD_DATA, ANEMIA_DISABILITY_WEIGHTS
from vivarium_gates_iv_iron.constants import data_keys


class Hemoglobin:
    """
    class for hemoglobin utilities and calculations that in turn will be used to find anemia status for simulants.
    """
    def __init__(self):
        pass

    @property
    def name(self):
        return "hemoglobin"

    def setup(self, builder: Builder):
        self.randomness = builder.randomness.get_stream(self.name)
        self.columns_created = ["country", "hemoglobin_distribution_propensity", "hemoglobin_percentile"]
        # load data
        # TODO: do this s/location/country in the loader?
        mean = builder.data.load(data_keys.HEMOGLOBIN.MEAN).rename(columns={"location": "country"})
        stddev = builder.data.load(data_keys.HEMOGLOBIN.STANDARD_DEVIATION).rename(columns={"location": "country"})
        self.location_weights = builder.data.load(data_keys.POPULATION.PLW_LOCATION_WEIGHTS).rename(columns={"location": "country"})
        index_columns = ["country", "sex", "age_start", 'age_end', 'year_start', 'year_end']
        mean = mean.set_index(index_columns)["value"].rename("mean")
        stddev = stddev.set_index(index_columns)["value"].rename("stddev")
        distribution_parameters = pd.concat([mean, stddev], axis=1).reset_index()
        self.distribution_parameters = builder.value.register_value_producer("hemoglobin.exposure_parameters",
            source=builder.lookup.build_table(distribution_parameters, key_columns=["sex", "country"], parameter_columns=["age", "year"]),
                                                                             requires_columns=["age", "sex", "country"])
        self.hemoglobin = builder.value.register_value_producer("hemoglobin.exposure", source=self.hemoglobin_source,
                                                                requires_values=["hemoglobin.exposure_parameters"],
                                                                requires_streams=[self.name])

        self.thresholds = builder.lookup.build_table(HEMOGLOBIN_THRESHOLD_DATA,
                                                     key_columns=["sex", "pregnancy_status"],
                                                     parameter_columns=["age"])

        self.anemia_levels = builder.value.register_value_producer("anemia_levels", source=self.anemia_source,
                                                                   requires_values=["hemoglobin.exposure"])

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=self.columns_created,
                                                 requires_streams=[self.name])

        self.population_view = builder.population.get_view(self.columns_created)
        builder.event.register_listener("time_step", self.on_time_step)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pop_update = pd.DataFrame({"country": self.randomness.choice(pop_data.index,
                                                                     choices=self.location_weights.country.to_list(),
                                                                     p=self.location_weights.value.to_list(),
                                                                     additional_key="country"),
                                   "hemoglobin_distribution_propensity": self.randomness.get_draw(
                                       pop_data.index,
                                       additional_key="hemoglobin_distribution_propensity"),
                                   "hemoglobin_percentile": self.randomness.get_draw(pop_data.index, additional_key="hemoglobin_percentile")},
                                  index=pop_data.index)
        self.population_view.update(pop_update)

    def on_time_step(self, event):
        self.distribution_parameters(event.index)
        self.hemoglobin(event.index)
        self.anemia_levels(event.index)
        return

    def hemoglobin_source(self, idx: pd.Index) -> pd.Series:
        distribution_parameters = self.distribution_parameters(idx)
        pop = self.population_view.get(idx)
        return self.sample_from_hemoglobin_distribution(pop["hemoglobin_distribution_propensity"], pop["hemoglobin_percentile"], distribution_parameters)

    def anemia_source(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.hemoglobin(index)
        thresholds = self.thresholds(index)
        choice_index = (hemoglobin_level.values[np.newaxis].T < thresholds).sum(axis=1)

        return pd.Series(np.array(["none", "mild", "moderate", "severe"])[choice_index], index=index, name="anemia_levels")

    @staticmethod
    def _gamma_ppf(propensity, mean, sd):
        """Returns the quantile for the given quantile rank (`propensity`) of a Gamma
        distribution with the specified mean and standard deviation.
        """
        shape = (mean / sd) ** 2
        scale = sd ** 2 / mean
        return scipy.stats.gamma(a=shape, scale=scale).ppf(propensity)

    @staticmethod
    def _mirrored_gumbel_ppf(propensity, mean, sd):
        """Returns the quantile for the given quantile rank (`propensity`) of a mirrored Gumbel
        distribution with the specified mean and standard deviation.
        """
        _alpha = HEMOGLOBIN_DISTRIBUTION_PARAMETERS.XMAX - mean \
                    - (sd * HEMOGLOBIN_DISTRIBUTION_PARAMETERS.EULERS_CONSTANT * np.sqrt(6) / np.pi)
        scale = sd * np.sqrt(6) / np.pi
        tmp = _alpha + (scale * HEMOGLOBIN_DISTRIBUTION_PARAMETERS.EULERS_CONSTANT)
        alpha = _alpha + HEMOGLOBIN_DISTRIBUTION_PARAMETERS.XMAX - (2 * tmp)
        return scipy.stats.gumbel_r(alpha, scale=scale).ppf(propensity)

    def sample_from_hemoglobin_distribution(self, propensity_distribution, propensity, exposure_parameters):
        """
        Returns a sample from an ensemble distribution with the specified mean and
        standard deviation (stored in `exposure_parameters`) that is 40% Gamma and
        60% mirrored Gumbel. The sampled value is a function of the two propensities
        `prop_dist` (used to choose whether to sample from the Gamma distribution or
        the mirrored Gumbel distribution) and `propensity` (used as the quantile rank
        for the selected distribution).
        """

        exposure_data = exposure_parameters
        mean = exposure_data["mean"]
        sd = exposure_data["stddev"]

        gamma = propensity_distribution < 0.4
        gumbel = ~gamma
        ret_val = pd.Series(index=propensity_distribution.index, name="value")
        ret_val.loc[gamma] = self._gamma_ppf(propensity, mean, sd)[gamma]
        ret_val.loc[gumbel] = self._mirrored_gumbel_ppf(propensity, mean, sd)[gumbel]
        return ret_val

    def disability_weight(self, index: pd.Index) -> pd.Series:
        hemoglobin_level = self.hemoglobin(index)
        thresholds = self.thresholds(index)
        choice_index = (hemoglobin_level.values[np.newaxis].T < thresholds).sum(axis=1)
        anemia_levels = pd.Series(np.array(["none", "mild", "moderate", "severe"])[choice_index], index=index,
                                 name="anemia_levels")
        return anemia_levels.map(ANEMIA_DISABILITY_WEIGHTS)


