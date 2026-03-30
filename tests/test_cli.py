from pathlib import Path

from typer.testing import CliRunner

from ai_leader.cli import app

runner = CliRunner()


def test_validate_data_command():
    path = Path("tests/fixtures/routing_eval_sample_small.csv")
    result = runner.invoke(app, ["validate-data", str(path)])
    assert result.exit_code == 0
    assert "Validated dataset" in result.stdout


def test_notebook_smoke_command():
    path = Path("tests/fixtures/routing_eval_sample_small.csv")
    result = runner.invoke(app, ["notebook-smoke", str(path)])
    assert result.exit_code == 0
