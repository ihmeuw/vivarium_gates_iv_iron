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
    PREGNANT_PREVALENCE: TargetString = TargetString(
        "cause.pregnancy.pregnant_prevalence"
    )
    NOT_PREGNANT_PREVALENCE: TargetString = TargetString(
        "cause.pregnancy.not_pregnant_prevalence"
    )
    POSTPARTUM_PREVALENCE: TargetString = TargetString(
        "cause.pregnancy.postpartum_prevalence"
    )
    INCIDENCE_RATE: TargetString = TargetString(
        "cause.pregnancy.incidence_rate"
    )
    ASFR: TargetString = TargetString('covariate.age_specific_fertility_rate.estimate'
    )
    SBR: TargetString = TargetString('covariate.stillbirth_to_live_birth_ratio.estimate'
    )
    INCIDENCE_RATE_ECTOPIC: TargetString = TargetString("cause.ectopic_pregnancy.incidence_rate"
                                                        )
    INCIDENCE_RATE_MISCARRIAGE: TargetString = TargetString("cause.maternal_abortion_and_miscarriage.incidence_rate"
                                                            )

    @property
    def name(self):
        return "pregnancy"

    @property
    def log_name(self):
        return "pregnancy"


PREGNANCY = __Pregnancy()


class __MaternalDisorders(NamedTuple):

    CSMR: TargetString = TargetString(
        "cause.maternal_disorders.cause_specific_mortality_rate"
    )
    INCIDENCE_RATE: TargetString = TargetString(
        "cause.maternal_disorders.incidence_rate"
    )
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.maternal_disorders.disability_weight"
    )

    @property
    def name(self):
        return "maternal_disorders"

    @property
    def log_name(self):
        return "maternal_disorders"


MATERNAL_DISORDERS = __MaternalDisorders()


class __MaternalHemorrhage(NamedTuple):

    CSMR: TargetString = TargetString(
        "cause.maternal_hemorrhage.cause_specific_mortality_rate"
    )
    INCIDENCE_RATE: TargetString = TargetString(
        "cause.maternal_hemorrhage.incidence_rate"
    )

    @property
    def name(self):
        return "maternal_hemorrhage"

    @property
    def log_name(self):
        return "maternal_hemorrhage"


MATERNAL_HEMORRHAGE = __MaternalHemorrhage()


class __LowBirthWeightShortGestation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.exposure'
    DISTRIBUTION: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.distribution'
    CATEGORIES: TargetString = 'risk_factor.low_birth_weight_and_short_gestation.categories'

    @property
    def name(self):
        return 'low_birth_weight_and_short_gestation'

    @property
    def log_name(self):
        return 'low birth weight and short gestation'


LBWSG = __LowBirthWeightShortGestation()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    PREGNANCY,
    PREGNANCY_OUTCOMES,
    MATERNAL_DISORDERS,
    MATERNAL_HEMORRHAGE,
    LBWSG
]
