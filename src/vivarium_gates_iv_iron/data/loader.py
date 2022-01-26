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
import pandas as pd

from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
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
        data_keys.PREGNANCY.PREGNANCY_INCIDENCE_RATE: load_pregnancy_incidence_rate,
        data_keys.PREGNANCY.INCIDENCE_C995: load_standard_data,
        data_keys.PREGNANCY.INCIDENCE_C374: load_standard_data,
        data_keys.PREGNANCY.ASFR: load_asfr,
        data_keys.PREGNANCY.SBR: load_sbr,
        data_keys.LBWSG.DISTRIBUTION: load_metadata,
        data_keys.LBWSG.CATEGORIES: load_metadata,
        data_keys.LBWSG.EXPOSURE: load_lbwsg_exposure,
        # TODO - add appropriate mappings
        # data_keys.DIARRHEA_PREVALENCE: load_standard_data,
        # data_keys.DIARRHEA_INCIDENCE_RATE: load_standard_data,
        # data_keys.DIARRHEA_REMISSION_RATE: load_standard_data,
        # data_keys.DIARRHEA_CAUSE_SPECIFIC_MORTALITY_RATE: load_standard_data,
        # data_keys.DIARRHEA_EXCESS_MORTALITY_RATE: load_standard_data,
        # data_keys.DIARRHEA_DISABILITY_WEIGHT: load_standard_data,
        # data_keys.DIARRHEA_RESTRICTIONS: load_metadata,
    }
    return mapping[lookup_key](lookup_key, location)


def load_population_location(key: str, location: str) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    if location == "LMICs":
        world_bank_1 = filter_population(interface.get_population_structure("World Bank Low Income"))
        world_bank_2 = filter_population(interface.get_population_structure("World Bank Lower Middle Income"))
        population_structure = pd.concat([world_bank_1, world_bank_2])
    else:
        population_structure = filter_population(interface.get_population_structure(location))
    return population_structure


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'location': location,
            'sex': 'Female',
            'age_start': 7,
            'age_end': 54,
            'year_start': 2021,
            'year_end': 2022,
        }
    ]).set_index(['location', 'sex', 'age_start', 'age_end', 'year_start', 'year_end'])
    # return interface.get_demographic_dimensions(location)


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


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


# TODO - add project-specific data functions here
def load_pregnancy_incidence_rate(key: str, location: str):
    # This is incidence_p = (ASFR + ASFR * SBR + incidence_c995 + incidence_c374) / prevalence_np
    #TODO make function to actually do this
    pass


def load_incidence(key: str, location: str):
    #TODO write function to load incidence for c995 and c374
    pass


def load_asfr(key: str, location: str):
    #asfr = utilities.get_data(key, entity, location, gbd_constants.SOURCES.EXPOSURE, 'rei_id',
    #                         metadata.AGE_GROUP.GBD_2019_LBWSG_EXPOSURE, metadata.GBD_2019_ROUND_ID, 'step4')
    # Regional, get_covariate_estimates: decomp_step=’step4’ or ‘iterative’ for GBD 2019, ‘step3’ or ‘iterative’ for GBD 2020
    pass


def load_sbr(key: str, location: str):
    #TODO implement
    # get_covariate_estimates: decomp_step =’step4’ or ‘iterative’ for GBD 2019, ‘step3’ or ‘iterative’ for GBD 2020
    pass


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


def filter_population(unfiltered: pd.DataFrame) -> pd.DataFrame:
    unfiltered = unfiltered.reset_index()
    filtered_pop = unfiltered[(unfiltered.sex == "Female") & (unfiltered.age_start >= 5) & (unfiltered.age_end <= 60)]
    filtered_pop = filtered_pop.set_index(metadata.ARTIFACT_INDEX_COLUMNS)

    return filtered_pop


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


# def get_prevalence_not_pregnant(key: str, location: str) -> pd.DataFrame:
#     np_prevalence = 1 - get_prevalence_pregnant() - get_prevalence_postpartum()
#
#     return np_prevalence
#
#
# def get_prevalence_pregnant(key: str, location: str) -> pd.DataFrame:
#     pregnancy_prevalence = (load_asfr() + load_asfr() * load_sbr() + incidence:c995 + incidence:c374) * 40/52
#
#     return pregnancy_prevalence
#
#
# def get_prevalence_postpartum(key: str, location: str) -> pd.DataFrame:
#     postpartum_prevalence = (load_asfr()+ load_asfr() * load_sbr() + incidence:c995 + incidence:c374) * 6/52
#
#     return postpartum_prevalence
