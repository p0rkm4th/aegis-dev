import json
from pathlib import Path


def test_validation_script_and_ci_workflow_exist():
    root = Path(__file__).parents[1]
    script = root / "scripts" / "validate.sh"
    workflow = root / ".github" / "workflows" / "ci.yml"
    assert script.is_file()
    assert workflow.is_file()
    assert "python -m pytest" in script.read_text()
    assert "bash scripts/validate.sh" in workflow.read_text()
    assert (root / "scripts" / "smoke_install.sh").is_file()
    sbom = json.loads((root / "provenance" / "SBOM.json").read_text())
    assert sbom["bomFormat"] == "CycloneDX"
    assert any(component["name"] == "pydantic" for component in sbom["components"])


def test_alpha_launcher_has_environment_fallback_and_actionable_failure():
    root = Path(__file__).parents[1]
    launcher = (root / "scripts" / "aegis").read_text()

    assert "AEGIS_PYTHON" in launcher
    assert "python3" in launcher
    assert 'PYTHONPATH="$repo_root/src' in launcher
    assert "could not find a usable Python environment" in launcher
