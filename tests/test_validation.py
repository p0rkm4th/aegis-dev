from pathlib import Path


def test_validation_script_and_ci_workflow_exist():
    root = Path(__file__).parents[1]
    script = root / "scripts" / "validate.sh"
    workflow = root / ".github" / "workflows" / "ci.yml"
    assert script.is_file()
    assert workflow.is_file()
    assert "python -m pytest" in script.read_text()
    assert "bash scripts/validate.sh" in workflow.read_text()
