"""
Extract Pareto-optimal configs from an Optuna study and output
ann-benchmarks config.yml run_groups.

Usage:
    python tuning/extract_pareto_configs.py \
        --db results/studies/glove200.db \
        --study glove200_multiprobe_pareto
"""

import argparse
from collections import defaultdict

import optuna
import yaml


def extract_pareto_configs(db_path: str, study_name: str, recall_targets=None):
    """Load Pareto front from Optuna DB and return run_groups for config.yml.

    If recall_targets is provided, selects one representative config per
    recall band (the trial closest to each target). This keeps run_groups
    manageable (~10 instead of ~30).
    """
    if recall_targets is None:
        recall_targets = [0.99, 0.89, 0.79, 0.69, 0.59, 0.49, 0.39, 0.29, 0.19, 0.09]

    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    best_trials = study.best_trials
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto-optimal trials: {len(best_trials)}")

    # Sort by recall descending
    best_trials.sort(key=lambda t: t.user_attrs.get("recall_at_k", 0), reverse=True)

    # Print the full Pareto front
    print(f"\n{'Recall@10':<12} {'Time(ms)':<10} {'Speedup':<10} | {'d':<4} {'g':<4} {'probe':<6}")
    print("-" * 55)
    for t in best_trials:
        rec = t.user_attrs.get("recall_at_k", 0)
        time_ms = t.user_attrs.get("avg_time_ms", 0)
        speedup = t.user_attrs.get("speedup", 0)
        p = t.params
        print(f"{rec:<12.3f} {time_ms:<10.3f} {speedup:<10.1f} | {p['pca_dims']:<4} {p['grid_size']:<4} {p['n_probe']:<6}")

    # Select one representative trial per recall band
    selected = []
    used_targets = []
    for target in recall_targets:
        closest = min(
            best_trials,
            key=lambda t: abs(t.user_attrs.get("recall_at_k", 0) - target),
        )
        # Avoid duplicates if two targets map to the same trial
        if closest not in selected:
            selected.append(closest)
            used_targets.append(target)

    print(f"\nSelected {len(selected)} representative configs for recall targets {recall_targets}:")
    print(f"{'Target':<8} {'Actual':<10} {'Time(ms)':<10} | {'d':<4} {'g':<4} {'probe':<6}")
    print("-" * 50)
    for target, t in zip(used_targets, selected):
        rec = t.user_attrs.get("recall_at_k", 0)
        time_ms = t.user_attrs.get("avg_time_ms", 0)
        p = t.params
        print(f"{target:<8.2f} {rec:<10.3f} {time_ms:<10.3f} | {p['pca_dims']:<4} {p['grid_size']:<4} {p['n_probe']:<6}")

    # Group selected trials by (pca_dims, grid_size) → collect n_probe values
    build_groups = defaultdict(set)
    for t in selected:
        p = t.params
        key = (p["pca_dims"], p["grid_size"])
        build_groups[key].add(p["n_probe"])

    # For each build group, fill n_probe with powers of 2 for smooth curve
    run_groups = {}
    for (dims, splits), probes in sorted(build_groups.items()):
        max_probe = max(probes)
        filled = set(probes)
        p = 1
        while p <= max_probe:
            filled.add(p)
            p *= 2
        filled.add(max_probe)

        group_name = f"multiprobe-d{dims}-g{splits}"
        run_groups[group_name] = {
            "args": [[dims], [splits]],
            "query_args": [sorted(filled)],
        }

    return run_groups


def format_config_yml(run_groups: dict) -> str:
    """Format run_groups into the full config.yml structure."""
    config = {
        "float": {
            "euclidean": [{
                "base_args": ["@metric"],
                "constructor": "AnnMultiprobe",
                "docker_tag": "ann-benchmarks-ann_multiprobe",
                "module": "ann_benchmarks.algorithms.ann_multiprobe",
                "name": "ann-multiprobe",
                "run_groups": run_groups,
            }],
            "angular": [{
                "base_args": ["@metric"],
                "constructor": "AnnMultiprobe",
                "docker_tag": "ann-benchmarks-ann_multiprobe",
                "module": "ann_benchmarks.algorithms.ann_multiprobe",
                "name": "ann-multiprobe",
                "run_groups": run_groups,
            }],
        }
    }
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Extract Pareto configs from Optuna DB")
    parser.add_argument("--db", required=True, help="Path to Optuna SQLite DB")
    parser.add_argument("--study", required=True, help="Optuna study name")
    parser.add_argument("--output", default=None, help="Output config.yml path (default: print to stdout)")
    args = parser.parse_args()

    run_groups = extract_pareto_configs(args.db, args.study)

    print(f"\n{'='*55}")
    print(f"Generated {len(run_groups)} run_groups:")
    for name, rg in run_groups.items():
        dims, splits = rg["args"][0][0], rg["args"][1][0]
        probes = rg["query_args"][0]
        print(f"  {name}: dims={dims}, splits={splits}, n_probe={probes}")

    yml = format_config_yml(run_groups)

    if args.output:
        with open(args.output, "w") as f:
            f.write(yml)
        print(f"\nWritten to {args.output}")
    else:
        print(f"\n{'='*55}")
        print("config.yml content:\n")
        print(yml)


if __name__ == "__main__":
    main()
