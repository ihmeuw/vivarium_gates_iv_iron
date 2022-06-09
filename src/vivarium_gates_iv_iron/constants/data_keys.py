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


class __Pregnancy(NamedTuple):
    ASFR: str = 'covariate.age_specific_fertility_rate.estimate'
    SBR: str = 'covariate.stillbirth_to_live_birth_ratio.estimate'
    INCIDENCE_RATE_MISCARRIAGE: str = "cause.maternal_abortion_and_miscarriage.incidence_rate"
    INCIDENCE_RATE_ECTOPIC: str = "cause.ectopic_pregnancy.incidence_rate"
    PREVALENCE: str = "covariate.pregnancy_prevalence.estimate"
    CONCEPTION_RATE: str = "covariate.conception_rate.estimate"
    CHILD_OUTCOME_PROBABILITIES: str = 'covariate.child_outcome_probability.estimate'
    PREGNANT_LACTATING_WOMEN_LOCATION_WEIGHTS: str = (
        "covariate.pregnant_and_lactating_women_location_weights.estimate"
    )
    WOMEN_REPRODUCTIVE_AGE_LOCATION_WEIGHTS: str = (
        "covariate.women_of_reproductive_age_location_weights.estimate"
    )
    HEMOGLOBIN_CORRECTION_FACTORS: str = 'covariate.hemoglobin_correction_factors.estimate'

    @property
    def name(self):
        return "pregnancy"

    @property
    def log_name(self):
        return "pregnancy"


PREGNANCY = __Pregnancy()


class __MaternalDisorders(NamedTuple):
    TOTAL_CSMR: str = "cause.maternal_disorders.cause_specific_mortality_rate"
    TOTAL_INCIDENCE_RATE: str = "cause.maternal_disorders.incidence_rate"
    HEMORRHAGE_CSMR: str = "cause.maternal_hemorrhage.cause_specific_mortality_rate"
    HEMORRHAGE_INCIDENCE_RATE: str = "cause.maternal_hemorrhage.incidence_rate"
    YLDS: str = "cause.maternal_disorders.ylds"

    PROBABILITY_FATAL: str = 'covariate.probability_fatal_maternal_disorder.estimate'
    PROBABILITY_NONFATAL: str = 'covariate.probability_nonfatal_maternal_disorder.estimate'
    PROBABILITY_HEMORRHAGE: str = "covariate.probability_maternal_hemorrhage.estimate"

    RR_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN: str = 'risk_factor.hemoglobin_on_maternal_hemorrhage.relative_risk'
    PAF_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN: str = 'risk_factor.hemoglobin_on_maternal_hemorrhage.paf'

    RR_MATERNAL_DISORDER_ATTRIBUTABLE_TO_HEMOGLOBIN: str = "risk_factor.hemoglobin_on_maternal_disorder.relative_risk"
    PAF_MATERNAL_DISORDER_ATTRIBUTABLE_TO_HEMOGLOBIN: str = "risk_factor.hemoglobin_on_maternal_disorder.paf"

    @property
    def name(self):
        return "maternal_disorders"

    @property
    def log_name(self):
        return "maternal_disorders"


MATERNAL_DISORDERS = __MaternalDisorders()


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
    EXPOSURE: str = 'risk_factor.low_birth_weight_and_short_gestation.exposure'
    DISTRIBUTION: str = 'risk_factor.low_birth_weight_and_short_gestation.distribution'
    CATEGORIES: str = 'risk_factor.low_birth_weight_and_short_gestation.categories'

    @property
    def name(self):
        return 'low_birth_weight_and_short_gestation'

    @property
    def log_name(self):
        return 'low birth weight and short gestation'


LBWSG = __LowBirthWeightShortGestation()


class __MaternalBMI(NamedTuple):
    PREVALENCE_LOW_BMI_ANEMIC: str = 'maternal_bmi.prevalance_low_bmi_anemic'
    PREVALENCE_LOW_BMI_NON_ANEMIC: str = 'maternal_bmi.prevalence_low_bmi_non_anemic'

    @property
    def name(self):
        return 'maternal_bmi'

    @property
    def log_name(self):
        return 'maternal BMI'


MATERNAL_BMI = __MaternalBMI()


class __MaternalInterventions(NamedTuple):
    COVERAGE: str = 'maternal_interventions.coverage'

    @property
    def name(self):
        return 'maternal_interventions'

    @property
    def log_name(self):
        return 'maternal interventions'


MATERNAL_INTERVENTIONS = __MaternalInterventions()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    PREGNANCY,
    MATERNAL_DISORDERS,
    LBWSG,
    HEMOGLOBIN,
    MATERNAL_BMI,
    MATERNAL_INTERVENTIONS,
]
