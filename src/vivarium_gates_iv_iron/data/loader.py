"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.

"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from gbd_mapping import causes, covariates, risk_factors, sequelae
from vivarium_gbd_access.utilities import get_draws
from vivarium.framework.artifact import EntityKey
from vivarium.framework.randomness import get_hash
from vivarium_gbd_access import constants as gbd_constants, gbd
from vivarium_inputs import (
    globals as vi_globals,
    interface,
    utilities as vi_utils,
    utility_data,
)
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_iv_iron.constants import data_keys, metadata
from vivarium_gates_iv_iron.data import utilities
from vivarium_gates_iv_iron.paths import (
    PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70_CSV as HGB_BELOW_70_CSV
)
from vivarium_gates_iv_iron.utilities import create_draws


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_standard_data,
        data_keys.PREGNANCY.ASFR: load_asfr,
        data_keys.PREGNANCY.SBR: load_sbr,

        data_keys.PREGNANCY.INCIDENCE_RATE: load_pregnancy_incidence_rate,
        data_keys.PREGNANCY.PREGNANT_PREVALENCE: get_prevalence_pregnant,
        data_keys.PREGNANCY.NOT_PREGNANT_PREVALENCE: get_prevalence_not_pregnant,
        data_keys.PREGNANCY.POSTPARTUM_PREVALENCE: get_prevalence_postpartum,
        data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE: load_standard_data,
        data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC: load_standard_data,
        data_keys.PREGNANCY.PREGNANT_LACTATING_WOMEN_LOCATION_WEIGHTS: get_pregnant_lactating_women_location_weights,
        data_keys.PREGNANCY.WOMEN_REPRODUCTIVE_AGE_LOCATION_WEIGHTS: get_women_reproductive_age_location_weights,
        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,
        data_keys.PREGNANCY_OUTCOMES.STILLBIRTH: load_pregnancy_outcome,
        data_keys.PREGNANCY_OUTCOMES.LIVE_BIRTH: load_pregnancy_outcome,
        data_keys.PREGNANCY_OUTCOMES.OTHER: load_pregnancy_outcome,
        data_keys.MATERNAL_DISORDERS.CSMR: load_standard_data,
        data_keys.MATERNAL_DISORDERS.INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_DISORDERS.YLDS: load_maternal_disorders_ylds,
        data_keys.MATERNAL_HEMORRHAGE.CSMR: load_standard_data,
        data_keys.MATERNAL_HEMORRHAGE.INCIDENCE_RATE: load_standard_data,
        data_keys.HEMOGLOBIN.MEAN: get_hemoglobin_data,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: get_hemoglobin_data,
        data_keys.HEMOGLOBIN.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70: get_hemoglobin_csv_data
    }
    return mapping[lookup_key](lookup_key, location)


###############################
# Generic loaders and helpers #
###############################

def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    return interface.get_measure(entity, key.measure, location).droplevel("location")


def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, "to_dict"):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


###################
# Population data #
###################


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    if location == "LMICs":
        world_bank_1 = interface.get_population_structure("World Bank Low Income")
        world_bank_2 = interface.get_population_structure("World Bank Lower Middle Income")
        population_structure = pd.concat([world_bank_1, world_bank_2])
    else:
        population_structure = interface.get_population_structure(location)
    return population_structure


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location)


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


##################
# Pregnancy Data #
##################

@lru_cache
def load_asfr(key: str, location: str):
    asfr = load_standard_data(key, location)
    asfr = asfr.reset_index()
    asfr_pivot = asfr.pivot(
        index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
        columns='parameter',
        values='value'
    )
    seed = f'{key}_{location}'
    asfr_draws = generate_lognormal_draws(asfr_pivot, seed)
    return asfr_draws


@lru_cache
def load_sbr(key: str, location: str):
    births_per_location_year, sbr = [], []
    for child_loc in utilities.get_child_locs(location):
        child_pop = get_data(data_keys.POPULATION.STRUCTURE, child_loc)
        child_asfr = get_data(data_keys.PREGNANCY.ASFR, child_loc)
        child_asfr.index = child_pop.index  # Add location back
        child_births = child_asfr.multiply(child_pop.value, axis=0).groupby(
            ['location', 'year_start']).sum()
        births_per_location_year.append(child_births)

        child_sbr = load_standard_data(data_keys.PREGNANCY.SBR, child_loc)
        child_sbr = (child_sbr
                     .reset_index(level='year_end', drop=True)
                     .reorder_levels(['parameter', 'year_start'])
                     .loc['mean_value']
                     .reindex(child_births.index, level='year_start'))
        sbr.append(child_sbr)

    births_per_location_year = pd.concat(births_per_location_year)
    sbr = pd.concat(sbr)

    births_per_year = births_per_location_year.groupby('year_start').transform('sum')
    sbr = (births_per_location_year
           .multiply(sbr.value, axis=0)
           .divide(births_per_year)
           .groupby('year_start')
           .sum()
           .reset_index())
    sbr['year_end'] = sbr['year_start'] + 1
    sbr = sbr.set_index(['year_start', 'year_end'])

    return sbr


def get_pregnant_lactating_women_location_weights(key: str, location: str):
    #  WRA * (ASFR + (ASFR * SBR) + incidence_c996 + incidence_c374)
    #      - divide by regional population
    plw_location_weights = pd.DataFrame()

    child_locs = utilities.get_child_locs(location)

    for loc in child_locs:
        # ASFR
        asfr = load_standard_data(data_keys.PREGNANCY.ASFR, loc)
        asfr = asfr.reset_index()
        asfr_pivot = asfr.pivot(index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
                                columns='parameter', values='value')
        asfr_draws = asfr_pivot.apply(create_draws, args=(data_keys.PREGNANCY.ASFR, loc), axis=1)

        # SBR
        sbr = load_standard_data(data_keys.PREGNANCY.SBR, loc)
        sbr = sbr.reset_index()
        sbr = sbr[sbr.parameter == "mean_value"]
        sbr = sbr.drop(["parameter"], axis=1).set_index(['year_start', 'year_end'])
        sbr = sbr.reset_index(level="year_end", drop=True).reindex(asfr_draws.index, level="year_start").value

        # WRA
        wra = get_wra(loc)
        wra = wra.reset_index().set_index("age_start").wra.reindex(asfr_draws.index, level="age_start").fillna(0)
        wra.loc["Male"] = 0

        incidence_c996 = load_standard_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, loc)
        incidence_c374 = load_standard_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, loc)

        # do math to calculate population of PLW
        plw_loc = (asfr_draws.mul(1 + sbr, axis=0) + incidence_c996 + incidence_c374).mul(wra, axis=0).groupby(["year_start", "year_end"]).sum()
        plw_loc = pd.concat([plw_loc.query("year_start==2019")], keys=[loc], names=['location'])
        plw_loc = plw_loc.droplevel(["year_start", "year_end"])

        plw_location_weights = plw_location_weights.append(plw_loc)

    # Divide each location by total region population
    plw_location_weights = plw_location_weights/plw_location_weights.sum(axis=0)
    return plw_location_weights



def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(
        data, interval_column="age", split_column_prefix="age"
    )
    data = vi_utils.split_interval(
        data, interval_column="year", split_column_prefix="year"
    )
    return vi_utils.sort_hierarchical_data(data)


def load_pregnancy_incidence_rate(key: str, location: str):
    not_pregnant = get_prevalence_not_pregnant(key, location)
    pregnancy_incidence_rate = _get_pregnancy_outcome_denominator(key, location) / not_pregnant

    return pregnancy_incidence_rate


def get_wra(location: str, decomp: str = "step4"):
    from db_queries import get_population
    location_id = utility_data.get_location_id(location)
    wra = get_population(
        location_id=location_id,
        age_group_id=[7, 8, 9, 10, 11, 12, 13, 14, 15],
        sex_id=2,
        gbd_round_id=metadata.GBD_2019_ROUND_ID,
        decomp_step=decomp,
    )

    # reshape to vivarium format
    wra = wra.set_index(['age_group_id', 'location_id', 'sex_id', 'year_id']).drop('run_id', axis=1)
    wra = utilities.scrub_gbd_conventions(wra, location)
    wra = vi_utils.split_interval(wra, interval_column='age', split_column_prefix='age')
    wra = vi_utils.split_interval(wra, interval_column='year', split_column_prefix='year')
    wra = vi_utils.sort_hierarchical_data(wra)

    wra = wra.rename({'population': 'wra'}, axis=1)
    wra.index = wra.index.droplevel('location')

    return wra


def get_entity(key: str):
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
        "alternative_risk_factor": alternative_risk_factors,
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]


def load_lbwsg_exposure(key: str, location: str) -> pd.DataFrame:
    if key != data_keys.LBWSG.EXPOSURE:
        raise ValueError(f'Unrecognized key {key}')

    key = EntityKey(key)
    entity = utilities.get_entity(key)
    data = utilities.get_data(key, entity, location, gbd_constants.SOURCES.EXPOSURE, 'rei_id',
                              metadata.AGE_GROUP.GBD_2019_LBWSG_EXPOSURE, metadata.GBD_2019_ROUND_ID, 'step4')
    data = data[data['year_id'] == 2019].drop(columns='year_id')
    data = utilities.process_exposure(data, key, entity, location, metadata.GBD_2019_ROUND_ID,
                                      metadata.AGE_GROUP.GBD_2019_LBWSG_EXPOSURE | metadata.AGE_GROUP.GBD_2020)
    data = data[data.index.get_level_values('year_start') == 2019]
    return data


def get_prevalence_not_pregnant(key: str, location: str) -> pd.DataFrame:
    np_prevalence = 1 - get_prevalence_pregnant(key, location) - get_prevalence_postpartum(key, location)

    return np_prevalence


def get_prevalence_pregnant(key: str, location: str) -> pd.DataFrame:
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)

    prevalence_pregnant = (((asfr + asfr * sbr) * 40 / 52) +
                           ((incidence_c995 + incidence_c374) * 24 / 52))

    return prevalence_pregnant


def get_prevalence_postpartum(key: str, location: str) -> pd.DataFrame:
    return _get_pregnancy_outcome_denominator(key, location) * 6 / 52


def _get_pregnancy_outcome_denominator(key: str, location: str, asfr=None, sbr=None,
                                       incidence_c995=None, incidence_c374=None):
    # ASFR + ASFR * SBR + incidence_c995 + incidence_c374)
    if asfr is None:
        asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    if sbr is None:
        sbr = get_data(data_keys.PREGNANCY.SBR, location)
    if incidence_c995 is None:
        incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    if incidence_c374 is None:
        incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)

    return asfr + asfr * sbr + incidence_c995 + incidence_c374


def load_pregnancy_outcome(key: str, location: str):
    # live_birht =  asfr/denom
    # stillbirth =  asfr*sbr
    # other = addition both incidence
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)
    pregnancy_denominator = _get_pregnancy_outcome_denominator(key, location, asfr=asfr, sbr=sbr,
                                                               incidence_c995=incidence_c995,
                                                               incidence_c374=incidence_c374)

    if key == data_keys.PREGNANCY_OUTCOMES.LIVE_BIRTH:
        return asfr / pregnancy_denominator
    elif key == data_keys.PREGNANCY_OUTCOMES.STILLBIRTH:
        return (asfr * sbr) / pregnancy_denominator
    elif key == data_keys.PREGNANCY_OUTCOMES.OTHER:
        return (incidence_c374 + incidence_c995) / pregnancy_denominator
    else:
        raise ValueError(f'Unrecognized key {key}')


def subset_to_wra(df):
    df = df.query("sex=='Female' & year_start==2019 & age_start >= 5 & age_end <= 60")

    return df


def reshape_to_vivarium_format(df, location):
    df = df.set_index(['age_group_id', 'sex_id', 'year_id'])
    df = utilities.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column='age', split_column_prefix='age')
    df = vi_utils.split_interval(df, interval_column='year', split_column_prefix='year')
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")

    return df


def get_maternal_ylds(entity_list, location):
    gbd_id_types = [entity.kind + '_id' for entity in entity_list]
    gbd_ids = [int(entity.gbd_id) for entity in entity_list]

    location_id = utility_data.get_location_id(location) if isinstance(location, str) else location

    ylds_draws = get_draws(
        gbd_id_types,
        gbd_ids,
        source=gbd_constants.SOURCES.COMO,
        year_id=2019,
        decomp_step=gbd_constants.DECOMP_STEP.STEP_5,
        gbd_round_id=gbd_constants.ROUND_IDS.GBD_2019,
        location_id=location_id,
        measure_id=vi_globals.MEASURES['YLDs']
    )

    groupby_cols = ['age_group_id', 'sex_id', 'year_id']
    draw_cols = [f"draw_{i}" for i in range(1000)]

    # aggregate by summing if given multiple entities
    if len(entity_list) > 1:
        ylds_draws = ylds_draws.groupby(groupby_cols)[draw_cols].sum().reset_index()

    return ylds_draws[groupby_cols + draw_cols]


def load_maternal_disorders_ylds(key: str, location: str) -> pd.DataFrame:
    # YLDS updated equation 4/14: (maternal_ylds - anemia_ylds) /
    #   (maternal_incidence - (acmr - csmr) * maternal_incidence - csmr)

    maternal_disorders = [causes.maternal_disorders]

    maternal_ylds = get_maternal_ylds(maternal_disorders, location)
    maternal_ylds = reshape_to_vivarium_format(maternal_ylds, location)

    anemia_sequelae = [sequelae.mild_anemia_due_to_maternal_hemorrhage,
                       sequelae.moderate_anemia_due_to_maternal_hemorrhage,
                       sequelae.severe_anemia_due_to_maternal_hemorrhage]

    anemia_ylds = get_maternal_ylds(anemia_sequelae, location)
    anemia_ylds = reshape_to_vivarium_format(anemia_ylds, location)

    maternal_incidence = get_data(data_keys.MATERNAL_DISORDERS.INCIDENCE_RATE, location)
    # Update incidence for 55-59 year age group to match 50-54 year age group
    maternal_incidence.iloc[-1] = maternal_incidence.iloc[-2]

    csmr = get_data(data_keys.MATERNAL_DISORDERS.CSMR, location)
    acmr = get_data(data_keys.POPULATION.ACMR, location)

    # TODO: replace nans with 0 here instead of in pregnancy component?
    return (maternal_ylds - anemia_ylds) / (maternal_incidence - (acmr - csmr) * maternal_incidence - csmr)


def get_hemoglobin_data(key: str, location: str):
    country_dfs = []
    child_locs = utilities.get_child_locs(location)

    for loc in child_locs:
        location_id = utility_data.get_location_id(loc) if isinstance(loc, str) else loc
        if key == data_keys.HEMOGLOBIN.MEAN:
            me_id = 10487
        elif key == data_keys.HEMOGLOBIN.STANDARD_DEVIATION:
            me_id = 10488
        else:
            raise KeyError("Invalid Hemoglobin key")

        hemoglobin_data = gbd.get_modelable_entity_draws(me_id=me_id, location_id=location_id)
        hemoglobin_data = reshape_to_vivarium_format(hemoglobin_data, loc)
        hemoglobin_data = pd.concat([hemoglobin_data], keys=[loc], names=['location'])
        country_dfs.append(hemoglobin_data)

    national_level_hemoglobin_data = pd.concat(country_dfs)

    return national_level_hemoglobin_data


def get_women_reproductive_age_location_weights(key: str, location: str):
    # Used to get location weights with women of reproductive age
    # This will be used instead of pregnant and lactating women location weights unless specified otherwise

    wra_location_weights = pd.DataFrame()
    child_locs = utilities.get_child_locs(location)

    for loc in child_locs:
        wra = get_wra(loc)

        wra_loc = wra
        wra_loc = pd.concat([wra_loc.query("year_start==2019")], keys=[loc], names=['location'])
        wra_location_weights = wra_location_weights.append(wra_loc)

        # Divide each location by total region population
    wra_location_weights = wra_location_weights / wra_location_weights.sum(axis=0)
    wra_location_weights = wra_location_weights.rename(columns={"wra": "value"})

    return wra_location_weights


def get_hemoglobin_csv_data(key: str, location: str):
    try:
        source_file = {
            data_keys.HEMOGLOBIN.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70: HGB_BELOW_70_CSV,
        }[key]
    except KeyError:
        raise ValueError(f'Unrecognized key {key}')
    try:
        location_id = utility_data.get_location_id(location)
    except KeyError:
        raise ValueError(f"Unrecognized location {location}")

    data = pd.read_csv(source_file)
    data = data[data["location_id"] == location_id]
    data = data[["draw", "age_group_id", "value"]]
    age_group_ids = data["age_group_id"].to_list()
    age_bins = utilities.get_gbd_age_bins(age_group_ids)
    data = data.merge(age_bins, on="age_group_id")
    data = data[["draw", "age_start", "age_end", "value"]]
    data = data.pivot(index=["age_start", "age_end"], columns='draw', values='value')
    return data


def generate_lognormal_draws(df, seed, quantiles=(0.025, 0.975)):
    mean = df['mean_value'].values
    lower = df['lower_value'].values
    upper = df['upper_value'].values
    assert np.all((lower <= mean) & (mean <= upper))
    assert np.all((lower == mean) == (upper == mean))

    sample_mask = (mean > 0) & (lower < mean) & (mean < upper)
    stdnorm_quantiles = stats.norm.ppf(quantiles)
    norm_quantiles = np.log([lower[sample_mask], upper[sample_mask]])
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
                stdnorm_quantiles[1] - stdnorm_quantiles[0])

    distribution = stats.lognorm(s=sigma, scale=mean[sample_mask])
    np.random.seed(get_hash(seed))
    lognorm_samples = distribution.rvs(size=(1000, sample_mask.sum())).T
    lognorm_samples = pd.DataFrame(lognorm_samples, index=df[sample_mask].index)

    use_means = np.tile(mean[~sample_mask], 1000).reshape((1000, ~sample_mask.sum())).T
    use_means = pd.DataFrame(use_means, index=df[~sample_mask].index)
    return pd.concat([lognorm_samples, use_means]).sort_index().rename(
        columns=lambda d: f'draw_{d}')