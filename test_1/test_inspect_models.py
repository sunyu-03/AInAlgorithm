from pathlib import Path

from inspect_models import add_validator_to_path, inspect_all


def test_all_six_models_load_and_slice():
    waam_root = Path.home() / "Desktop" / "WAAM"
    add_validator_to_path(waam_root / "TEAM_PROJECT" / "week1")
    records = inspect_all(Path.home() / "Desktop" / "WAAM_Validator-main" / "tests")
    assert [record["model"] for record in records] == ["01", "02", "03", "04", "05", "06"]
    assert all(record["mesh"]["watertight"] for record in records)
    assert all(record["layers"]["count"] > 0 for record in records)
    assert all(record["layers"]["count"] == record["layers"]["nonempty_count"] for record in records)

