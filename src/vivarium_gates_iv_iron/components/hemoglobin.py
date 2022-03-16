import numpy as np
import pandas as pd
import scipy

from vivarium_gates_iv_iron.constants.data_values import HEMOGLOBIN_DISTRIBUTION_PARAMETERS


class Hemoglobin:
    """
    class for hemoglobin utilities and calculations that in turn will be used to find anemia status for simulants.
    """
    def __init__(self):
        pass

    @property
    def name(self):
        return "hemoglobin"

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
        mean = exposure_data['mean']
        sd = exposure_data['sd']

        gamma = propensity_distribution < 0.4
        gumbel = ~gamma
        ret_val = pd.Series(index=propensity_distribution.index, name='value')
        ret_val.loc[gamma] = self._gamma_ppf(propensity.loc[gamma], mean, sd)
        ret_val.loc[gumbel] = self._mirrored_gumbel_ppf(propensity.loc[gumbel], mean, sd)
        return ret_val
