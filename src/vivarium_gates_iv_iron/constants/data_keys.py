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
    ASFR: str = 'covariate.age_specific_fertility_rate.estimate'
    SBR: str = 'covariate.stillbirth_to_live_birth_ratio.estimate'
    INCIDENCE_RATE_MISCARRIAGE: str = "cause.maternal_abortion_and_miscarriage.incidence_rate"
    INCIDENCE_RATE_ECTOPIC: str = "cause.ectopic_pregnancy.incidence_rate"
    PREVALENCE: str = "covariate.pregnancy_prevalence.estimate"
    CONCEPTION_RATE: str = "covariate.conception_rate.estimate"
    CHILD_OUTCOME_PROBABILITIES: str = 'covariate.child_outcome_probability.estimate'
    PROBABILITY_FATAL_MATERNAL_DISORDER: str = (
        'covariate.probability_fatal_maternal_disorder.estimate'
    )
    PROBABILITY_NONFATAL_MATERNAL_DISORDER: str = (
        'covariate.probability_nonfatal_maternal_disorder.estimate'
    )
    PROBABILITY_MATERNAL_HEMORRHAGE: str = (
        "covariate.probability_maternal_hemorrhage.estimate"
    )

    PREGNANT_LACTATING_WOMEN_LOCATION_WEIGHTS: str = (
        "pregnancy.pregnant_and_lactating_women_location_weights"
    )
    WOMEN_REPRODUCTIVE_AGE_LOCATION_WEIGHTS: str = (
        "pregnancy.women_of_reproductive_age_location_weights"
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
    YLDS: TargetString = TargetString(
        "cause.maternal_disorders.ylds"
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


class _Hemoglobin(NamedTuple):
    MEAN: TargetString = TargetString("risk_factor.hemoglobin.mean")
    STANDARD_DEVIATION: TargetString = TargetString("risk_factor.hemoglobin.standard_deviation")
    PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70: TargetString = TargetString(
        "risk_factor.hemoglobin.pregnant_proportion_below_70_gL")

    @property
    def name(self):
        return 'hemoglobin'

    @property
    def log_name(self):
        return 'hemoglobin'


HEMOGLOBIN = _Hemoglobin()


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
    LBWSG,
    HEMOGLOBIN,
]
