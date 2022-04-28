from pathlib import Path

import vivarium_gates_iv_iron
from vivarium_gates_iv_iron.constants import metadata

BASE_DIR = Path(vivarium_gates_iv_iron.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{metadata.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RESULTS_ROOT = Path(f"/share/costeffectiveness/results/{metadata.PROJECT_NAME}/")
CSV_RAW_DATA_ROOT = BASE_DIR / "data" / "raw_data"

# Proportion of pregnant women with hemoglobin less than 70 g/L
PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70_CSV = (
        CSV_RAW_DATA_ROOT / "pregnant_proportion_with_hgb_below_70_age_specific.csv")
