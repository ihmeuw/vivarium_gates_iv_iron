from pathlib import Path

import vivarium_gates_iv_iron
from vivarium_gates_iv_iron.constants import metadata

BASE_DIR = Path(vivarium_gates_iv_iron.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{metadata.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RESULTS_ROOT = Path(f"/share/costeffectiveness/results/{metadata.PROJECT_NAME}/")

# todo fix a better output location
#CHILD_DATA_OUTPUT_DIR = RESULTS_ROOT / "child_data"
CHILD_DATA_OUTPUT_DIR = Path("/Users/mkappel/viv_out") / "child_data"

