from typing import Union

from gbd_mapping import (
    causes,
    covariates,
    risk_factors,
    ModelableEntity,
)
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import (
    globals as vi_globals,
    utilities as vi_utils,
    utility_data,
)
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_gates_iv_iron.data import extra_gbd


def get_entity(key: Union[str, EntityKey]) -> ModelableEntity:
    key = EntityKey(key)
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors,
    }
    return type_map[key.type][key.name]


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
