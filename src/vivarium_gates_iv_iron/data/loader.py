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
from vivarium.framework.artifact import EntityKey
from vivarium.framework.randomness import get_hash
from vivarium_gbd_access import gbd
from vivarium_inputs import (
    globals as vi_globals,
    interface,
    utilities as vi_utils,
    utility_data,
)

from vivarium_gates_iv_iron import paths
from vivarium_gates_iv_iron.constants import (
    data_keys,
    data_values,
    metadata,
    models,
)
from vivarium_gates_iv_iron.data import (
    extra_gbd,
    sampling,
)
from vivarium_gates_iv_iron.data.utilities import get_entity


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
        data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE: load_standard_data,
        data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC: load_standard_data,
        data_keys.PREGNANCY.PREVALENCE: load_pregnancy_prevalence,
        data_keys.PREGNANCY.CONCEPTION_RATE: load_conception_rate,
        data_keys.PREGNANCY.CHILD_OUTCOME_PROBABILITIES: load_child_outcome_probabilities,
        data_keys.PREGNANCY.PREGNANT_LACTATING_WOMEN_LOCATION_WEIGHTS: load_pregnant_lactating_women_location_weights,
        data_keys.PREGNANCY.WOMEN_REPRODUCTIVE_AGE_LOCATION_WEIGHTS: load_women_reproductive_age_location_weights,
        data_keys.PREGNANCY.HEMOGLOBIN_CORRECTION_FACTORS: load_pregnancy_hemoglobin_correction,

        data_keys.MATERNAL_DISORDERS.TOTAL_CSMR: load_standard_data,
        data_keys.MATERNAL_DISORDERS.TOTAL_INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_DISORDERS.HEMORRHAGE_CSMR: load_standard_data,
        data_keys.MATERNAL_DISORDERS.HEMORRHAGE_INCIDENCE_RATE: load_standard_data,
        data_keys.MATERNAL_DISORDERS.YLDS: load_maternal_disorders_ylds,
        data_keys.MATERNAL_DISORDERS.PROBABILITY_FATAL: load_probability_fatal_maternal_disorder,
        data_keys.MATERNAL_DISORDERS.PROBABILITY_NONFATAL: load_probability_nonfatal_maternal_disorder,
        data_keys.MATERNAL_DISORDERS.PROBABILITY_HEMORRHAGE: load_probability_maternal_hemorrhage,

        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,

        data_keys.HEMOGLOBIN.MEAN: get_hemoglobin_data,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: get_hemoglobin_data,
        data_keys.HEMOGLOBIN.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70: get_hemoglobin_csv_data,

        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_ANEMIC: load_bmi_prevalence,
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_NON_ANEMIC: load_bmi_prevalence,

        data_keys.MATERNAL_INTERVENTIONS.COVERAGE: load_intervention_coverage,
    }
    return mapping[lookup_key](lookup_key, location)


###############################
# Generic loaders and helpers #
###############################

@lru_cache
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


def load_asfr(key: str, location: str):
    asfr = load_standard_data(key, location)
    asfr = asfr.reset_index()
    asfr_pivot = asfr.pivot(
        index=[col for col in metadata.ARTIFACT_INDEX_COLUMNS if col != "location"],
        columns='parameter',
        values='value'
    )
    seed = f'{key}_{location}'
    asfr_draws = sampling.generate_vectorized_lognormal_draws(asfr_pivot, seed)
    return asfr_draws


def load_sbr(key: str, location: str):
    try:
        return get_child_sbr(location)
    except vi_globals.DataDoesNotExistError:
        pass

    births_per_location_year, sbr = [], []
    for child_loc in get_child_locs(location):
        child_pop = get_data(data_keys.POPULATION.STRUCTURE, child_loc)
        child_asfr = get_data(data_keys.PREGNANCY.ASFR, child_loc)
        child_asfr.index = child_pop.index  # Add location back
        child_births = (child_asfr
                        .multiply(child_pop.value, axis=0)
                        .groupby(['location', 'year_start'])
                        .sum())
        births_per_location_year.append(child_births)

        child_sbr = get_child_sbr(child_loc)
        child_sbr = (child_sbr
                     .reset_index(level='year_end', drop=True)
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


def get_child_sbr(location: str):
    child_sbr = load_standard_data(data_keys.PREGNANCY.SBR, location)
    child_sbr = (child_sbr
                 .reorder_levels(['parameter', 'year_start', 'year_end'])
                 .loc['mean_value'])
    return child_sbr


def load_pregnancy_prevalence(key: str, location: str):
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)
    pregnancy_end_rate = asfr + asfr * sbr + incidence_c995 + incidence_c374

    pregnant_prevalence = (
        data_values.DURATIONS.FULL_TERM * (asfr + asfr * sbr)
        + data_values.DURATIONS.PARTIAL_TERM * (incidence_c995 + incidence_c374)
    )
    pregnant_prevalence['pregnancy_status'] = models.PREGNANT_STATE
    maternal_disorders_prevalence = pd.DataFrame(
        0., index=pregnant_prevalence.index, columns=pregnant_prevalence.columns
    )
    maternal_disorders_prevalence['pregnancy_status'] = models.MATERNAL_DISORDER_STATE
    no_maternal_disorders_prevalence = (
        data_values.DURATIONS.PREPOSTPARTUM * pregnancy_end_rate
    )
    no_maternal_disorders_prevalence['pregnancy_status'] = models.NO_MATERNAL_DISORDER_STATE
    postpartum_prevalence = (
        data_values.DURATIONS.POSTPARTUM * pregnancy_end_rate
    )
    postpartum_prevalence['pregnancy_status'] = models.POSTPARTUM_STATE

    prevalence = pd.concat([
        pregnant_prevalence,
        maternal_disorders_prevalence,
        no_maternal_disorders_prevalence,
        postpartum_prevalence,
    ])
    not_pregnant_prevalence = 1 - prevalence.groupby(prevalence.index.names).sum()
    not_pregnant_prevalence['pregnancy_status'] = models.NOT_PREGNANT_STATE
    prevalence = pd.concat([prevalence, not_pregnant_prevalence])
    prevalence = prevalence.set_index('pregnancy_status', append=True)

    return prevalence


def _get_pregnancy_end_rate(location: str):
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)
    pregnancy_end_rate = (asfr + asfr * sbr + incidence_c995 + incidence_c374)
    return pregnancy_end_rate.reorder_levels(asfr.index.names)


def _get_not_pregnant_prevalence(location: str):
    prevalence = get_data(data_keys.PREGNANCY.PREVALENCE, location)
    prevalence = prevalence.reset_index(level='pregnancy_status')
    not_pregnant = prevalence.pregnancy_status == models.NOT_PREGNANT_STATE
    return prevalence.loc[not_pregnant].drop(columns='pregnancy_status')


def load_conception_rate(key: str, location: str):
    not_pregnant_prevalence = _get_not_pregnant_prevalence(location)
    pregnancy_end_rate = _get_pregnancy_end_rate(location)
    conception_rate = pregnancy_end_rate / not_pregnant_prevalence
    return conception_rate


def load_child_outcome_probabilities(key: str, location: str):
    asfr = get_data(data_keys.PREGNANCY.ASFR, location)
    sbr = get_data(data_keys.PREGNANCY.SBR, location)
    incidence_c995 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_MISCARRIAGE, location)
    incidence_c374 = get_data(data_keys.PREGNANCY.INCIDENCE_RATE_ECTOPIC, location)
    pregnancy_end_rate = _get_pregnancy_end_rate(location)

    live_birth_probability = asfr / pregnancy_end_rate
    live_birth_probability['pregnancy_outcome'] = models.LIVE_BIRTH_OUTCOME
    still_birth_probability = (
        (asfr * sbr / pregnancy_end_rate).reorder_levels(asfr.index.names)
    )
    still_birth_probability['pregnancy_outcome'] = models.STILLBIRTH_OUTCOME
    other_probability = (incidence_c374 + incidence_c995) / pregnancy_end_rate
    other_probability['pregnancy_outcome'] = models.OTHER_OUTCOME

    probabilities = pd.concat([
        live_birth_probability,
        still_birth_probability,
        other_probability,
    ]).fillna(0)
    invalid_probability = 1 - probabilities.groupby(probabilities.index.names).sum()
    invalid_probability['pregnancy_outcome'] = models.INVALID_OUTCOME
    probabilities = pd.concat([probabilities, invalid_probability])
    probabilities = probabilities.set_index('pregnancy_outcome', append=True)

    return probabilities


def load_pregnant_lactating_women_location_weights(key: str, location: str):
    weights = []
    for child_loc in get_child_locs(location):
        child_pop = get_data(data_keys.POPULATION.STRUCTURE, child_loc)
        child_pregnancy_end_rate = _get_pregnancy_end_rate(location)
        child_pregnancy_end_rate.index = child_pop.index

        weight = (child_pregnancy_end_rate
                  .mul(child_pop.value, axis=0)
                  .groupby(["location", "year_start", "year_end"])
                  .sum())
        weights.append(weight)

    weights = pd.concat(weights)
    weights = weights / weights.groupby(["year_start", "year_end"]).transform("sum")
    return weights


def load_women_reproductive_age_location_weights(key: str, location: str):
    pops = pd.concat([
        get_data(data_keys.POPULATION.STRUCTURE, child_loc)
        for child_loc in get_child_locs(location)
    ])
    total_pop = (pops
                 .groupby([n for n in pops.index.names if n != 'location'])
                 .transform("sum"))
    weights = pops / total_pop
    return weights


def load_pregnancy_hemoglobin_correction(key: str, location: str):
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    data = []
    for pregnancy_status in models.PREGNANCY_MODEL_STATES:
        sample_status = (pregnancy_status if pregnancy_status == models.NOT_PREGNANT_STATE
                         else models.PREGNANT_STATE)
        params = data_values.HEMOGLOBIN_CORRECTION_FACTORS[sample_status]

        np.random.seed(get_hash(f'{key}_{sample_status}_{location}'))
        for param_name, param_set in zip(('mean', 'stddev'), params):
            dist = sampling.get_norm_from_quantiles(*param_set)
            samples = pd.DataFrame(
                np.tile(dist.rvs(size=1000), (len(demography), 1)),
                columns=vi_globals.DRAW_COLUMNS,
                index=demography.index,
            )
            samples['pregnancy_status'] = pregnancy_status
            samples['parameter'] = param_name
            data.append(samples)

    data = pd.concat(data).set_index(['pregnancy_status', 'parameter'], append=True)
    return data


###########################
# Maternal Disorders Data #
###########################

def load_maternal_disorders_ylds(key: str, location: str) -> pd.DataFrame:
    # YLDS updated equation 4/14: (maternal_ylds - anemia_ylds) /
    #   (maternal_incidence - (acmr - csmr) * maternal_incidence - csmr)
    groupby_cols = ['age_group_id', 'sex_id', 'year_id']
    draw_cols = [f"draw_{i}" for i in range(1000)]

    all_ylds = extra_gbd.get_maternal_disorder_ylds(location)
    all_ylds = all_ylds[groupby_cols + draw_cols]
    all_ylds = reshape_to_vivarium_format(all_ylds, location)

    anemia_ylds = extra_gbd.get_anemia_ylds(location)
    anemia_ylds = anemia_ylds.groupby(groupby_cols)[draw_cols].sum().reset_index()
    anemia_ylds = reshape_to_vivarium_format(anemia_ylds, location)

    acmr = get_data(data_keys.POPULATION.ACMR, location)
    csmr = get_data(data_keys.MATERNAL_DISORDERS.TOTAL_CSMR, location)
    incidence = get_data(data_keys.MATERNAL_DISORDERS.TOTAL_INCIDENCE_RATE, location)
    idx_cols = incidence.index.names
    incidence = incidence.reset_index()
    # FIXME: Why only here???
    #   Update incidence for 55-59 year age group to match 50-54 year age group
    to_duplicate = incidence.loc[(incidence.sex == 'Female') & (incidence.age_start == 50.0)]
    to_duplicate['age_start'] = 55.0
    to_duplicate['age_end'] = 60.0
    to_drop = incidence.loc[(incidence.sex == 'Female') & (incidence.age_start == 55.0)]
    incidence = pd.concat([
        incidence.drop(to_drop.index), to_duplicate
    ]).set_index(idx_cols).sort_index()

    ylds_per_case = (
        (all_ylds - anemia_ylds)
        / (incidence - (acmr - csmr) * incidence - csmr)
    ).fillna(0)
    return ylds_per_case


def load_probability_fatal_maternal_disorder(key: str, location: str):
    md_csmr = get_data(data_keys.MATERNAL_DISORDERS.TOTAL_CSMR, location)
    pregnancy_end_rate = _get_pregnancy_end_rate(location)
    probability = md_csmr / pregnancy_end_rate
    return probability.fillna(0.)


def load_probability_nonfatal_maternal_disorder(key: str, location: str):
    md_inc = get_data(data_keys.MATERNAL_DISORDERS.TOTAL_INCIDENCE_RATE, location)
    md_csmr = get_data(data_keys.MATERNAL_DISORDERS.TOTAL_CSMR, location)
    pregnancy_end_rate = _get_pregnancy_end_rate(location)
    probability = (md_inc - md_csmr) / pregnancy_end_rate
    return probability.fillna(0.)


def load_probability_maternal_hemorrhage(key: str, location: str):
    mh_inc = get_data(data_keys.MATERNAL_DISORDERS.HEMORRHAGE_INCIDENCE_RATE, location)
    mh_csmr = get_data(data_keys.MATERNAL_DISORDERS.HEMORRHAGE_CSMR, location)
    pregnancy_end_rate = _get_pregnancy_end_rate(location)

    p_any_hemorrhage = ((mh_inc - mh_csmr) / pregnancy_end_rate).fillna(0.)

    p_moderate_distribution = sampling.get_truncnorm_from_quantiles(
        *data_values.PROBABILITY_MODERATE_MATERNAL_HEMORRHAGE
    )
    np.random.seed(get_hash(f'{key}_{location}'))
    p_moderate_given_hemorrhage = pd.DataFrame(
        np.tile(p_moderate_distribution.rvs(size=1000), (len(p_any_hemorrhage), 1)),
        columns=p_any_hemorrhage.columns,
        index=p_any_hemorrhage.index
    )

    p_moderate = p_any_hemorrhage * p_moderate_given_hemorrhage
    p_moderate['hemorrhage_status'] = models.MODERATE_MATERNAL_HEMORRHAGE_STATE
    p_severe = p_any_hemorrhage * (1 - p_moderate_given_hemorrhage)
    p_severe['hemorrhage_status'] = models.SEVERE_MATERNAL_HEMORRHAGE_STATE
    p_none = 1 - p_any_hemorrhage
    p_none['hemorrhage_status'] = models.NOT_MATERNAL_HEMORRHAGE_STATE

    probability = pd.concat([p_moderate, p_severe, p_none])
    probability = probability.set_index('hemorrhage_status', append=True)

    probability = probability / probability.groupby(mh_inc.index.names).transform('sum')

    return probability


##############
# LBWSG Data #
##############

def load_lbwsg_exposure(key: str, location: str) -> pd.DataFrame:
    entity = get_entity(data_keys.LBWSG.EXPOSURE)
    data = extra_gbd.load_lbwsg_exposure(location)
    # This category was a mistake in GBD 2019, so drop.
    extra_residual_category = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
    data = data.loc[data['parameter'] != extra_residual_category]
    idx_cols = ['location_id', 'sex_id', 'parameter']
    data = data.set_index(idx_cols)[vi_globals.DRAW_COLUMNS]

    # Sometimes there are data values on the order of 10e-300 that cause
    # floating point headaches, so clip everything to reasonable values
    data = data.clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    # normalize so all categories sum to 1
    total_exposure = data.groupby(['location_id', 'sex_id']).transform('sum')
    data = (data / total_exposure).reset_index()
    data = reshape_to_vivarium_format(data, location)
    return data


###################
# Hemoglobin Data #
###################

def get_hemoglobin_data(key: str, location: str):
    me_id = {
        data_keys.HEMOGLOBIN.MEAN: 10487,
        data_keys.HEMOGLOBIN.STANDARD_DEVIATION: 10488
    }[key]

    country_dfs = []
    for child_loc in get_child_locs(location):
        location_id = utility_data.get_location_id(child_loc)
        hemoglobin_data = gbd.get_modelable_entity_draws(me_id=me_id, location_id=location_id)
        hemoglobin_data = reshape_to_vivarium_format(hemoglobin_data, child_loc)
        hemoglobin_data = pd.concat([hemoglobin_data], keys=[child_loc], names=['location'])
        country_dfs.append(hemoglobin_data)

    national_level_hemoglobin_data = pd.concat(country_dfs)
    return national_level_hemoglobin_data


def get_hemoglobin_csv_data(key: str, location: str):
    location_id = utility_data.get_location_id(location)
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    data = pd.read_csv(paths.PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70_CSV)
    data = data.set_index('location_id').loc[location_id]
    age_bins = utility_data.get_age_bins()
    data = data.merge(age_bins, on="age_group_id")
    data = data.pivot(index=["age_start", "age_end"], columns='draw', values='value')
    data = (data
            .reset_index(level='age_end', drop=True)
            .reindex(demography.index, level='age_start', fill_value=0.))
    return data


################
# Maternal BMI #
################

def load_bmi_prevalence(key: str, location: str):
    location_id = utility_data.get_location_id(location)
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)

    path = {
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_ANEMIC: paths.PREVALENCE_LOW_BMI_ANEMIC_CSV,
        data_keys.MATERNAL_BMI.PREVALENCE_LOW_BMI_NON_ANEMIC: paths.PREVALENCE_LOW_BMI_NON_ANEMIC_CSV,
    }[key]
    data = pd.read_csv(path)

    data = (data
            .set_index(['location_id', 'age_group_id', 'draw'])
            .loc[location_id]
            .value
            .unstack())
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(data, interval_column='age', split_column_prefix='age')
    data.index = data.index.droplevel(['location', 'age_end'])
    data = data.reindex(demography.index, level='age_start').fillna(0.)

    data = vi_utils.sort_hierarchical_data(data)
    data.index = data.index.droplevel("location")

    return data


##########################
# Maternal interventions #
##########################

def load_intervention_coverage(key: str, location: str) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)

    df = pd.read_csv(paths.MATERNAL_INTERVENTION_COVERAGE_CSV)
    df = df.drop(columns=['Unnamed: 0', 'scale_up'])
    df = df.set_index(['location_id', 'year', 'intervention', 'draw']).loc[location_id]

    dfs = []
    for scenario in metadata.SCENARIOS:
        data = df[df[f'{scenario}_scenario'] == 1].value.unstack()
        data.columns.name = None
        data['scenario'] = scenario
        data = data.reset_index().set_index(['scenario', 'year', 'intervention'])
        dfs.append(data)
    df = pd.concat(dfs).sort_index().reset_index()
    return df


###########
# Helpers #
###########

def get_child_locs(location):
    parent_id = utility_data.get_location_id(location)
    hierarchy = extra_gbd.get_gbd_hierarchy()

    is_child_loc = hierarchy.path_to_top_parent.str.contains(f',{parent_id},')
    is_country = hierarchy.location_type == "admin0"
    child_locs = hierarchy.loc[is_child_loc & is_country, 'location_name'].tolist()

    return child_locs


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=vi_globals.DRAW_COLUMNS)
    df = vi_utils.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column='age', split_column_prefix='age')
    df = vi_utils.split_interval(df, interval_column='year', split_column_prefix='year')
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")
    return df
