"""CLI-level smoke test for league-4p command."""

from pathlib import Path

from tarot.cli import _cmd_league_4p


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_cli_league_4p_runs_one_generation(tmp_path: Path):
    args = _Args(
        generations=1,
        population_size=4,
        rounds_per_generation=1,
        deals_per_match=1,
        elite_fraction=0.25,
        mutation_prob=0.5,
        mutation_std=0.1,
        seed=0,
        output_dir=str(tmp_path / "league_run"),
    )

    _cmd_league_4p(args)

    gen_file = tmp_path / "league_run" / "generation_000.json"
    assert gen_file.exists()

