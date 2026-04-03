"""Compare failure-monitor approaches across environments.

Usage:
    python experiments/compare_approaches.py

Re-evaluates each saved approach with its matching failure monitor
on 15 seeds, then prints solve rate, mean steps, and std of solve rate.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

RUNS = [
    {
        "env": "conveyorbelt",
        "load_dir": "outputs/2026-04-03/00-34-48",
    },
    {
        "env": "hovercraft",
        "load_dir": "outputs/2026-04-02/21-58-55",
    },
    {
        "env": "blocks",
        "load_dir": "outputs/2026-04-02/22-18-09",
    },
]

SEEDS = [0, 79, 158, 237, 317, 396, 475, 555, 634, 713, 793, 872, 951, 1031, 1110]
MAX_STEPS = 50000


def run_evaluation(env: str, load_dir: str, seed: int) -> Path:
    """Run evaluation for a single seed and return the results.json path."""
    cmd = [
        sys.executable,
        "experiments/run_experiment.py",
        "approach=agentic",
        f"environment={env}",
        f"failure_monitor={env}",
        f"approach.load_dir={load_dir}",
        f"seed={seed}",
        "num_eval_tasks=1",
        f"max_steps={MAX_STEPS}",
        "mcp_tools=[]",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-300:]}")

    # Find the most recently created results.json
    outputs = sorted(
        Path("outputs").rglob("results.json"), key=lambda p: p.stat().st_mtime
    )
    if not outputs:
        raise FileNotFoundError(f"No results.json found after running {env} seed={seed}")
    return outputs[-1]


def main() -> None:
    print("=" * 70)
    print(f"Comparing approaches across {len(RUNS)} environments")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)
    print()

    all_results = {}

    for run in RUNS:
        env = run["env"]
        load_dir = run["load_dir"]
        print(f"Evaluating {env} ({load_dir}) ...")

        solved_list = []
        steps_list = []

        for seed in SEEDS:
            print(f"  seed={seed}", end=" ", flush=True)
            results_path = run_evaluation(env, load_dir, seed)

            with open(results_path) as f:
                data = json.load(f)

            ep = data["per_episode"][0]
            solved_list.append(1.0 if ep["solved"] else 0.0)
            steps_list.append(ep["num_steps"])
            status = f"solved in {ep['num_steps']} steps" if ep["solved"] else f"FAILED ({ep['num_steps']} steps)"
            print(status)

        all_results[env] = {
            "solved": np.array(solved_list),
            "steps": np.array(steps_list, dtype=float),
        }
        print()

    # Print comparison table
    print("=" * 70)
    print(
        f"{'Environment':<15} {'Solve Rate':<12} {'Std (solve)':<12} "
        f"{'Mean Steps':<12} {'Std Steps':<12}"
    )
    print("-" * 70)
    for run in RUNS:
        env = run["env"]
        r = all_results[env]
        solve_rate = r["solved"].mean()
        solve_std = r["solved"].std()
        mean_steps = r["steps"].mean()
        std_steps = r["steps"].std()
        print(
            f"{env:<15} {solve_rate:<12.0%} {solve_std:<12.3f} "
            f"{mean_steps:<12.1f} {std_steps:<12.1f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
