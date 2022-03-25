import numpy as np
import pandas as pd
import scipy.stats

from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


from vivarium_gates_iv_iron.constants.data_values import HEMOGLOBIN_DISTRIBUTION_PARAMETERS
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
        mean = builder.data.load(data_keys.HEMOGLOBIN.MEAN)
        stddev = builder.data.load(data_keys.HEMOGLOBIN.STANDARD_DEVIATION)
        # TODO replace --->>>
        location_weights = mean.copy().drop(columns=["value"])
        # lookup data, categorical column "country"
        mean_dfs, stddev_dfs = [], []
        # TODO have this in artifact and correct the location weights
        for country in ["Bangladesh", "Pakistan", "India"]:
            loc_mean = mean.copy()
            loc_stddev = stddev.copy()
            loc_mean["country"] = country
            loc_stddev["country"] = country
            mean_dfs.append(loc_mean)
            stddev_dfs.append(loc_stddev)
            location_weights[country] = 1/3  # will be pop of country over pop of region
        index_columns = ["country", "sex", "age_start", 'age_end', 'year_start', 'year_end']
        mean = pd.concat(mean_dfs).set_index(index_columns)["value"].rename("mean")
        stddev = pd.concat(stddev_dfs).set_index(index_columns)["value"].rename("stddev")
        # <<<--- TODO replace
        distribution_parameters = pd.concat([mean, stddev], axis=1).reset_index()
        # TODO: look in Risk code to make sure we have canonical exposure naming
        self.distribution_parameters = builder.value.register_value_producer("hemoglobin.exposure_parameters",
            source=builder.lookup.build_table(distribution_parameters, key_columns=["sex", "country"], parameter_columns=["age", "year"]),
                                                                             requires_columns=["age", "sex", "country"])
        self.location_weights = builder.lookup.build_table(location_weights, key_columns=["sex"],
                                                           parameter_columns=["age", "year"])
        # TODO: canonical naming
        self.hemoglobin = builder.value.register_value_producer("hemoglobin", source=self.hemoglobin_source,
                                                                requires_values=["hemoglobin.exposure_parameters"],
                                                                requires_streams=[self.name])

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=self.columns_created,
                                                 requires_streams=[self.name])

        self.population_view = builder.population.get_view(self.columns_created)
        builder.event.register_listener("time_step", self.on_time_step)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        location_weights = self.location_weights(pop_data.index)
        pop_update = pd.DataFrame({"country": self.randomness.choice(pop_data.index, choices=location_weights.columns.tolist(),
                                                                     p=location_weights, additional_key="country"),
                                   "hemoglobin_distribution_propensity": self.randomness.get_draw(pop_data.index,
                                                                                     additional_key="hemoglobin_distribution_propensity"),
                                   "hemoglobin_percentile": self.randomness.get_draw(pop_data.index, additional_key="hemoglobin_percentile")},
                                  index=pop_data.index)
        self.population_view.update(pop_update)

    def on_time_step(self, event):
        self.distribution_parameters(event.index)
        self.hemoglobin(event.index)
        breakpoint()
        return

    def hemoglobin_source(self, idx: pd.Index) -> pd.Series:
        distribution_parameters = self.distribution_parameters(idx)
        pop = self.population_view.get(idx)
        return self.sample_from_hemoglobin_distribution(pop["hemoglobin_distribution_propensity"], pop["hemoglobin_percentile"], distribution_parameters)


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
        ret_val.loc[gamma] = self._gamma_ppf(propensity.loc[gamma], mean, sd)
        ret_val.loc[gumbel] = self._mirrored_gumbel_ppf(propensity.loc[gumbel], mean, sd)
        return ret_val
