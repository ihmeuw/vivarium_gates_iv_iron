import click
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Tuple
from pathlib import Path
from loguru import logger

from vivarium.framework.engine import Builder
from vivarium.framework.randomness import get_hash

from vivarium_gates_iv_iron.constants import metadata


def len_longest_location() -> int:
    """Returns the length of the longest location in the project.

    Returns
    -------
       Length of the longest location in the project.
    """
    return len(max(metadata.LOCATIONS, key=len))


def sanitize_location(location: str):
    """Cleans up location formatting for writing and reading from file names.

    Parameters
    ----------
    location
        The unsanitized location name.

    Returns
    -------
        The sanitized location name (lower-case with white-space and
        special characters removed.

    """
    # FIXME: Should make this a reversible transformation.
    return location.replace(" ", "_").replace("'", "_").lower()


def delete_if_exists(*paths: Union[Path, List[Path]], confirm=False):
    paths = paths[0] if isinstance(paths[0], list) else paths
    existing_paths = [p for p in paths if p.exists()]
    if existing_paths:
        if confirm:
            # Assumes all paths have the same root dir
            root = existing_paths[0].parent
            names = [p.name for p in existing_paths]
            click.confirm(
                f"Existing files {names} found in directory {root}. Do you want to delete and replace?",
                abort=True,
            )
        for p in existing_paths:
            logger.info(f"Deleting artifact at {str(p)}.")
            p.unlink()


def get_random_variable_draws(columns: pd.Index, seed: str, distribution) -> pd.Series:
    return pd.Series([get_random_variable(x, seed, distribution) for x in range(0, columns.size)], index=columns)


def get_random_variable(draw: int, seed: str, distribution) -> pd.Series:
    np.random.seed(get_hash(f'{seed}_draw_{draw}'))
    return distribution.rvs()


def get_random_variable_draws_for_location(columns: pd.Index, location: str, seed: str, distribution) -> pd.Series:
    return get_random_variable_draws(columns, f"{seed}_{location}", distribution)


def get_norm_from_quantiles(mean: float, lower: float, upper: float,
                            quantiles: Tuple[float, float] = (0.025, 0.975)) -> stats.norm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stats.norm(loc=mean, scale=sd)


def get_truncnorm_from_quantiles(mean: float, lower: float, upper: float,
                                 quantiles: Tuple[float, float] = (0.025, 0.975),
                                 lower_clip: float = 0.0, upper_clip: float = 1.0) -> stats.truncnorm:
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    sd = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    try:
        a = (lower_clip - mean) / sd
        b = (upper_clip - mean) / sd
        return stats.truncnorm(loc=mean, scale=sd, a=a, b=b)
    except ZeroDivisionError:
        # degenerate case: if upper == lower, then use the mean with sd==0
        return stats.norm(loc=mean, scale=sd)


def get_lognorm_from_quantiles(mean: float, lower: float, upper: float,
                               quantiles: Tuple[float, float] = (0.025, 0.975)) -> stats.lognorm:
    """Returns a frozen lognormal distribution with the specified mean, such that
    (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]).
    """
    # Let Y ~ norm(mu, sigma^2) and X = exp(Y), where mu = log(mean)
    # so X ~ lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
    # We will determine sigma from the two specified quantiles lower and upper.
    if not (lower <= mean <= upper):
        raise ValueError(
            f"The mean ({mean}) must be between the lower ({lower}) and upper ({upper}) "
            "quantile values."
        )
    try:
        # mean (and mean) of the normal random variable Y = log(X)
        mu = np.log(mean)
        # quantiles of the standard normal distribution corresponding to quantile_ranks
        stdnorm_quantiles = stats.norm.ppf(quantiles)
        # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
        norm_quantiles = np.log([lower, upper])
        # standard deviation of Y = log(X) computed from the above quantiles for Y
        # and the corresponding standard normal quantiles
        sigma = (norm_quantiles[1] - norm_quantiles[0]) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
        # Frozen lognormal distribution for X = exp(Y)
        # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the mean)
        return stats.lognorm(s=sigma, scale=mean)
    except:
        return stats.norm(loc=mean, scale=0)


def create_draws(df: pd.DataFrame, key: str, location: str, distribution_function=get_lognorm_from_quantiles):
    """
    Parameters
    ----------
    df: Multi-index dataframe with mean, lower, and upper values columns.
    location
    key:
    distribution_function: Distribution function to use to create draws
    Returns
    -------

    """
    # location defined in namespace outside of function
    mean = df['mean_value']
    lower = df['lower_value']
    upper = df['upper_value']

    Tuple = (key, distribution_function(mean=mean, lower=lower, upper=upper))
    # pull index from constants
    draws = get_random_variable_draws_for_location(pd.Index([f'draw_{i}' for i in range(0, 1000)]), location, *Tuple)

    return draws


def create_draw(draw: int, distribution_parameters: Tuple, key: str, location: str, distribution_function=get_lognorm_from_quantiles):
    """
    Parameters
    ----------
    draw: Input draw number
    distribution_parameters: Parameters for distribution (mean, lower upper)
    location: Location string, used in seed generation
    key: Key for the variable, used in seed generation
    distribution_function: Distribution function to use to create the draw
    Returns
    -------

    """
    distribution = distribution_function(mean=distribution_parameters[0],
                                         lower=distribution_parameters[1],
                                         upper=distribution_parameters[2])
    seed = f"{key}_{location}"
    return get_random_variable(draw, seed, distribution)


def load_and_unstack(builder: Builder, data_key: str, unstack_col: str) -> pd.DataFrame:
    data = builder.data.load(data_key)
    idx_cols = data.columns.difference(['value', unstack_col])
    data = data.pivot(index=idx_cols, columns=unstack_col, values='value')
    return data.reset_index()
