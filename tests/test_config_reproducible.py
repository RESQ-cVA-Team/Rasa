from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parents[1] / "src" / "core" / "config.yml"


def test_every_trained_component_has_a_seed() -> None:
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    trained = [step for step in config["pipeline"] + config["policies"] if "epochs" in step]
    assert len(trained) == 4
    for step in trained:
        assert "random_seed" in step, step["name"]
