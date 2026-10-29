"""Simulate a BioEmu2 trajectory and save it for the fly-through viewer.

Example:
    uv run python viewer/make_trajectory.py --pdb-id 1UBQ --steps 100 --lag-time-ns 1 \
        --out outputs/fly/ubiquitin
"""

import argparse
from pathlib import Path

from bioemu2 import System, load_model, sample_equilibrium, save_frames, simulate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdb-id", help="PDB entry to simulate.")
    source.add_argument("--system", type=Path, help="BioEmu2 system YAML file.")
    parser.add_argument("--steps", type=int, default=100, help="Number of propagation steps.")
    parser.add_argument("--lag-time-ns", type=float, default=1.0, help="Time between frames.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--device", default=None, help="Torch device, such as cpu; default cuda.")
    args = parser.parse_args()

    if args.pdb_id is not None:
        system = System.from_pdb_id(args.pdb_id)
    else:
        system = System.from_yaml(args.system)
    if args.device == "cpu":
        model = load_model(device="cpu", precision="32-true", matmul_precision=None)
    else:
        model = load_model(device=args.device)
    x0 = sample_equilibrium(system, model=model, num_samples=1, batch_size=1, seed=args.seed)
    (trajectory,) = simulate_trajectory(
        system,
        model=model,
        x0=x0,
        lag_time_ns=args.lag_time_ns,
        num_trajectory_steps=args.steps,
        batch_size=1,
        seed=args.seed + 1,
    )
    path = save_frames(trajectory, args.out)
    print(f"saved {trajectory.n_frames} frames of {trajectory.n_atoms} atoms to {path}")


if __name__ == "__main__":
    main()
