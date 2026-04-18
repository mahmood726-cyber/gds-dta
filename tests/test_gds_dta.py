import importlib.util
import json
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "simulation.py"
    spec = importlib.util.spec_from_file_location("gds_dta_simulation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_relative_outputs(tmp_path):
    module = load_module()

    result = module.main(seed=42, project_root=tmp_path)

    certification_path = tmp_path / "certification.json"
    data_path = tmp_path / "data.csv"

    assert certification_path.exists()
    assert data_path.exists()

    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert certification["project"] == "gds-dta"
    assert certification["methods"] == ["Moses", "EMS", "EWEF"]
    assert result["dataframe"].shape[0] == 40
