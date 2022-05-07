from numbers import Real
from typing import List, Set, Union
import warnings

import pandas as pd

from gbd_mapping import causes, covariates, risk_factors, ModelableEntity, RiskFactor
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import constants as gbd_constants, gbd
from vivarium_gbd_access.utilities import get_draws, query
from vivarium_inputs import globals as vi_globals, utilities as vi_utils, utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors, AlternativeRiskFactor
from vivarium_inputs.validation.raw import check_metadata

from vivarium_gates_iv_iron.constants.metadata import (
    AGE_GROUP,
    GBD_2019_ROUND_ID,
    GBD_2020_ROUND_ID,
)


def get_entity(key: Union[str, EntityKey]) -> ModelableEntity:
    key = EntityKey(key)
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    return type_map[key.type][key.name]


def get_child_locs(location):
    from db_queries import get_location_metadata
    # location_set_id 35 is for GBD model results
    hierarchy = get_location_metadata(
        location_set_id=35,
        decomp_step='step4',
        gbd_round_id=GBD_2019_ROUND_ID,
    )
    parent_id = utility_data.get_location_id(location)

    is_child_loc = hierarchy.path_to_top_parent.str.contains(f',{parent_id},')
    is_country = hierarchy.location_type == "admin0"
    child_locs = hierarchy.loc[is_child_loc & is_country, 'location_name'].tolist()

    return child_locs


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=vi_globals.DRAW_COLUMNS)
    df = scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column='age', split_column_prefix='age')
    df = vi_utils.split_interval(df, interval_column='year', split_column_prefix='year')
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")
    return df


def normalize_age_and_years(data: pd.DataFrame, fill_value: Real = None,
                            cols_to_fill: List[str] = vi_globals.DRAW_COLUMNS,
                            gbd_round_id: int = GBD_2020_ROUND_ID,
                            age_group_ids: List[int] = AGE_GROUP.GBD_2020) -> pd.DataFrame:
    data = vi_utils.normalize_sex(data, fill_value, cols_to_fill)

    # vi_inputs.normalize_year(data)
    binned_years = get_gbd_estimation_years(gbd_round_id)
    years = {'annual': list(range(min(binned_years), max(binned_years) + 1)), 'binned': binned_years}

    if 'year_id' not in data:
        # Data doesn't vary by year, so copy for each year.
        df = []
        for year in years['annual']:
            fill_data = data.copy()
            fill_data['year_id'] = year
            df.append(fill_data)
        data = pd.concat(df, ignore_index=True)
    elif set(data.year_id) == set(years['binned']):
        data = vi_utils.interpolate_year(data)
    else:  # set(data.year_id.unique()) == years['annual']
        pass

    # Dump extra data.
    data = data[data.year_id.isin(years['annual'])]

    data = _normalize_age(data, fill_value, cols_to_fill, age_group_ids)
    return data


def _normalize_age(data: pd.DataFrame, fill_value: Real, cols_to_fill: List[str],
                   age_group_ids: List[int] = None) -> pd.DataFrame:
    data_ages = set(data.age_group_id.unique()) if 'age_group_id' in data.columns else set()
    gbd_ages = set(utility_data.get_age_group_ids()) if not age_group_ids else set(age_group_ids)

    if not data_ages:
        # Data does not correspond to individuals, so no age column necessary.
        pass
    elif data_ages == {vi_globals.SPECIAL_AGES['all_ages']}:
        # Data applies to all ages, so copy.
        dfs = []
        for age in gbd_ages:
            missing = data.copy()
            missing.loc[:, 'age_group_id'] = age
            dfs.append(missing)
        data = pd.concat(dfs, ignore_index=True)
    elif data_ages < gbd_ages:
        # Data applies to subset, so fill other ages with fill value.
        key_columns = list(data.columns.difference(cols_to_fill))
        key_columns.remove('age_group_id')
        expected_index = pd.MultiIndex.from_product([data[c].unique() for c in key_columns] + [gbd_ages],
                                                    names=key_columns + ['age_group_id'])

        data = (data.set_index(key_columns + ['age_group_id'])
                .reindex(expected_index, fill_value=fill_value)
                .reset_index())
    else:  # data_ages == gbd_ages
        pass
    return data


def get_gbd_estimation_years(gbd_round_id: int) -> List[int]:
    """Gets the estimation years for a particular gbd round."""
    from db_queries import get_demographics
    warnings.filterwarnings("default", module="db_queries")

    return get_demographics(gbd_constants.CONN_DEFS.EPI, gbd_round_id=gbd_round_id)['year_id']


def scrub_gbd_conventions(data: pd.DataFrame, location: str, age_group_ids: List[int] = None) -> pd.DataFrame:
    data = vi_utils.scrub_location(data, location)
    data = vi_utils.scrub_sex(data)
    data = _scrub_age(data, age_group_ids)
    data = vi_utils.scrub_year(data)
    data = vi_utils.scrub_affected_entity(data)
    return data


def process_exposure(
    data: pd.DataFrame,
    entity: Union[RiskFactor, AlternativeRiskFactor],
    location: str,
    gbd_round_id: int,
) -> pd.DataFrame:
    data['rei_id'] = entity.gbd_id

    # from vivarium_inputs.extract.extract_exposure
    allowable_measures = [vi_globals.MEASURES['Proportion'], vi_globals.MEASURES['Continuous'],
                          vi_globals.MEASURES['Prevalence']]
    proper_measure_id = set(data.measure_id).intersection(allowable_measures)
    if len(proper_measure_id) != 1:
        raise vi_globals.DataAbnormalError(f'Exposure data have {len(proper_measure_id)} measure id(s). '
                                           f'Data should have exactly one id out of {allowable_measures} '
                                           f'but came back with {proper_measure_id}.')
    data = data[data.measure_id == proper_measure_id.pop()]

    # from vivarium_inputs.core.get_exposure
    data = data.drop('modelable_entity_id', 'columns')

    if entity.name in vi_globals.EXTRA_RESIDUAL_CATEGORY:
        # noinspection PyUnusedLocal
        cat = vi_globals.EXTRA_RESIDUAL_CATEGORY[entity.name]
        data = data.drop(labels=data.query('parameter == @cat').index)
        data[vi_globals.DRAW_COLUMNS] = data[vi_globals.DRAW_COLUMNS].clip(lower=vi_globals.MINIMUM_EXPOSURE_VALUE)

    if entity.distribution in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
        tmrel_cat = utility_data.get_tmrel_category(entity)
        exposed = data[data.parameter != tmrel_cat]
        unexposed = data[data.parameter == tmrel_cat]
        #  FIXME: We fill 1 as exposure of tmrel category, which is not correct.
        data = pd.concat([normalize_age_and_years(exposed, fill_value=0, gbd_round_id=gbd_round_id),
                          normalize_age_and_years(unexposed, fill_value=1, gbd_round_id=gbd_round_id)],
                         ignore_index=True)

        # normalize so all categories sum to 1
        cols = list(set(data.columns).difference(vi_globals.DRAW_COLUMNS + ['parameter']))
        data = data.set_index(cols + ['parameter'])
        sums = (
            data.groupby(cols)[vi_globals.DRAW_COLUMNS].sum()
                .reindex(index=data.index)
        )
        data = data.divide(sums).reset_index()
    else:
        data = vi_utils.normalize(data, fill_value=0)

    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS + ['parameter'])
    data = reshape_to_vivarium_format(data, location)
    return data


def _scrub_age(data: pd.DataFrame, age_group_ids: List[int] = None) -> pd.DataFrame:
    if 'age_group_id' in data.index.names:
        age_bins = get_gbd_age_bins(age_group_ids).set_index('age_group_id')
        id_levels = data.index.levels[data.index.names.index('age_group_id')]
        interval_levels = [pd.Interval(age_bins.age_start[age_id], age_bins.age_end[age_id], closed='left')
                           for age_id in id_levels]
        data.index = data.index.rename(names='age', level='age_group_id').set_levels(levels=interval_levels, level='age')
    return data


def get_gbd_age_bins(age_group_ids: List[int] = None) -> pd.DataFrame:
    # If no age group ids are specified, use the standard GBD 2019 age bins
    if not age_group_ids:
        age_group_ids = gbd.get_age_group_id()
    # from gbd.get_age_bins()
    q = f"""
                SELECT age_group_id,
                       age_group_years_start,
                       age_group_years_end,
                       age_group_name
                FROM age_group
                WHERE age_group_id IN ({','.join([str(a) for a in age_group_ids])})
                """
    raw_age_bins = query(q, 'shared')

    # from utility_data.get_age_bins()
    age_bins = (
        raw_age_bins[['age_group_id', 'age_group_name', 'age_group_years_start', 'age_group_years_end']]
        .rename(columns={'age_group_years_start': 'age_start', 'age_group_years_end': 'age_end'})
    )

    # set age start for birth prevalence age bin to -1 to avoid validation issues
    age_bins.loc[age_bins['age_end'] == 0.0, 'age_start'] = -1.0
    return age_bins
