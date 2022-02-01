from typing import NamedTuple

from vivarium_public_health.utilities import TargetString


#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    AGE_BINS: str = "population.age_bins"
    DEMOGRAPHY: str = "population.demographic_dimensions"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __PregnancyOutcomes(NamedTuple):
    # Live birth, still birth, other
    STILLBIRTH: str = "pregnancy_outcomes.stillbirth"
    LIVE_BIRTH: str = "pregnancy_outcomes.live_birth"
    OTHER: str = "pregnancy_outcomes.other"  # (abortion, miscarriage, ectopic pregnancy)

    @property
    def name(self):
        return "pregnancy_outcomes"

    @property
    def log_name(self):
        return "pregnancy_outcomes"


PREGNANCY_OUTCOMES = __PregnancyOutcomes()


class __Pregnancy(NamedTuple):

    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREGNANCY_PREVALENCE: TargetString = TargetString(
        "cause.pregnancy.prevalence"
    )
    PREGNANCY_INCIDENCE_RATE: TargetString = TargetString(
        "cause.pregnancy.incidence_rate"
    )
    # PREGNANCY_REMISSION_RATE: TargetString = TargetString(
    #     "cause.pregnancy.remission_rate"
    # )
    # DISABILITY_WEIGHT: TargetString = TargetString(
    #     "cause.pregnancy.disability_weight"
    # )
    ASFR: TargetString = TargetString('covariate.age_specific_fertility_rate.estimate'
    )
    SBR: TargetString = TargetString('covariate.stillbirth_to_live_birth_ratio.estimate'
    )
    INCIDENCE_C374: TargetString = TargetString("cause.ectopic_pregnancy.incidence_rate"
    )
    INCIDENCE_C995: TargetString = TargetString("cause.maternal_abortion_and_miscarriage.incidence_rate"
    )
    # EMR: TargetString = TargetString("cause.pregnancy.excess_mortality_rate")
    # CSMR: TargetString = TargetString(
    #     "cause.pregnancy.cause_specific_mortality_rate"
    # )
    # RESTRICTIONS: TargetString = TargetString("cause.pregnancy.restrictions")
    #
    # # Useful keys not for the artifact - distinguished by not using the colon type declaration
    # RAW_DISEASE_PREVALENCE = TargetString("sequela.raw_disease.prevalence")
    # RAW_DISEASE_INCIDENCE_RATE = TargetString("sequela.raw_disease.incidence_rate")

    @property
    def name(self):
        return "pregnancy"

    @property
    def log_name(self):
        return "pregnancy"


PREGNANCY = __Pregnancy()

MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    # TODO: list all key groups here
    PREGNANCY
]


class __LowBirthWeightShortGestation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.exposure'
    DISTRIBUTION: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.distribution'
    CATEGORIES: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.categories'
    #RELATIVE_RISK: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.relative_risk'
    #RELATIVE_RISK_INTERPOLATOR: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.relative_risk_interpolator'

    #PAF: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.population_attributable_fraction'

    # Useful keys not for the artifact - distinguished by not using the colon type declaration
    #BIRTH_WEIGHT_EXPOSURE = TargetString('risk_factor.low_birth_weight.birth_exposure')

    @property
    def name(self):
        return 'low_birth_weight_and_short_gestation'

    @property
    def log_name(self):
        return 'low birth weight and short gestation'


LBWSG = __LowBirthWeightShortGestation()