import optuna
import yaml

from tuning.extract_pareto_configs import extract_pareto_configs, format_config_yml

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DISTS = {
    "pca_dims": optuna.distributions.IntDistribution(2, 8),
    "grid_size": optuna.distributions.IntDistribution(2, 15),
    "n_probe": optuna.distributions.IntDistribution(1, 256),
}


def _trial(recall, time_ms, d, g, probe):
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=[1.0 - recall, time_ms],
        params={"pca_dims": d, "grid_size": g, "n_probe": probe},
        distributions=_DISTS,
        user_attrs={"recall_at_k": recall, "avg_time_ms": time_ms, "speedup": 1.0},
    )


def _make_study(tmp_path):
    db = tmp_path / "study.db"
    study = optuna.create_study(
        study_name="t",
        storage=f"sqlite:///{db}",
        directions=["minimize", "minimize"],
    )
    # All non-dominated (recall and time rise together).
    study.add_trial(_trial(0.90, 1.0, 6, 2, 4))
    study.add_trial(_trial(0.80, 0.5, 6, 2, 2))
    study.add_trial(_trial(0.70, 0.3, 6, 4, 2))
    study.add_trial(_trial(0.50, 0.1, 7, 2, 8))
    return str(db)


def test_extract_groups_by_dims_splits_and_fills_powers_of_two(tmp_path):
    db = _make_study(tmp_path)
    run_groups = extract_pareto_configs(db, "t")

    assert set(run_groups) == {"multiprobe-d6-g2", "multiprobe-d6-g4", "multiprobe-d7-g2"}

    g62 = run_groups["multiprobe-d6-g2"]
    assert g62["args"] == [[6], [2]]
    assert g62["query_args"] == [[1, 2, 4]]            # {2,4} filled with powers of 2

    assert run_groups["multiprobe-d6-g4"]["query_args"] == [[1, 2]]
    assert run_groups["multiprobe-d7-g2"]["query_args"] == [[1, 2, 4, 8]]


def test_format_config_yml_structure_and_docker_tag():
    run_groups = {"multiprobe-d6-g2": {"args": [[6], [2]], "query_args": [[1, 2, 4]]}}
    parsed = yaml.safe_load(format_config_yml(run_groups))

    for metric in ("euclidean", "angular"):
        block = parsed["float"][metric][0]
        assert block["docker_tag"] == "ann-benchmarks-ann_multiprobe"
        assert block["name"] == "ann-multiprobe"
        assert "multiprobe-d6-g2" in block["run_groups"]
