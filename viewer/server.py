"""Serve BioEmu2 trajectories to the fly-through viewer page.

Each trajectory directory holds ``topology.cif`` and ``samples.xtc``, as written by
``save_frames``. Frames are superposed on frame 0, centred, and sent in ångström.

Example:
    uv run python viewer/server.py outputs/fly/ubiquitin outputs/fly/chignolin
"""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bioemu2  # noqa: F401  (registers the mdtraj CIF patches)
import mdtraj
import numpy as np

VIEWER_DIR = Path(__file__).parent
BOND_CUTOFF_NM = 0.19


def _bonds(trajectory: mdtraj.Trajectory) -> np.ndarray:
    """Topology bonds, with distance-based bonds for any atom the topology leaves unbonded."""
    topology = trajectory.topology
    if topology.n_bonds == 0:
        topology.create_standard_bonds()
    pairs = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds}
    bonded = {i for pair in pairs for i in pair}
    loose = [atom.index for atom in topology.atoms if atom.index not in bonded]
    if loose:
        xyz = trajectory.xyz[0]
        for i in loose:
            distances = np.linalg.norm(xyz - xyz[i], axis=1)
            for j in np.flatnonzero((distances < BOND_CUTOFF_NM) & (distances > 0)):
                pairs.add(tuple(sorted((i, int(j)))))
    return np.array(sorted(pairs), dtype=np.int32).reshape(-1, 2)


def load_trajectory(directory: Path) -> dict:
    trajectory = mdtraj.load(str(directory / "samples.xtc"), top=str(directory / "topology.cif"))
    alignment = trajectory.topology.select("name CA")
    trajectory.superpose(trajectory, 0, atom_indices=alignment if len(alignment) >= 3 else None)
    coords = (trajectory.xyz - trajectory.xyz[0].mean(axis=0)) * 10.0
    meta = {
        "name": directory.name,
        "n_atoms": trajectory.n_atoms,
        "n_frames": trajectory.n_frames,
        "times_ps": trajectory.time.tolist(),
        "elements": [
            atom.element.symbol if atom.element is not None else "X"
            for atom in trajectory.topology.atoms
        ],
        "bonds": _bonds(trajectory).ravel().tolist(),
    }
    return {"meta": meta, "coords": coords.astype("<f4").tobytes()}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, trajectories: list[dict], **kwargs) -> None:
        self.trajectories = trajectories
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def do_GET(self) -> None:
        parts = self.path.split("?")[0].strip("/").split("/")
        if parts[0] != "api":
            return super().do_GET()
        if parts[1:] == ["list"]:
            return self._send(
                json.dumps([t["meta"]["name"] for t in self.trajectories]).encode(),
                "application/json",
            )
        if len(parts) == 4 and parts[1] == "traj" and parts[2].isdigit():
            index = int(parts[2])
            if index < len(self.trajectories):
                trajectory = self.trajectories[index]
                if parts[3] == "meta":
                    return self._send(json.dumps(trajectory["meta"]).encode(), "application/json")
                if parts[3] == "coords":
                    return self._send(trajectory["coords"], "application/octet-stream")
        self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", type=Path, nargs="+", help="Trajectory directories.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    trajectories = [load_trajectory(directory) for directory in args.directories]
    for trajectory in trajectories:
        meta = trajectory["meta"]
        print(f"{meta['name']}: {meta['n_atoms']} atoms, {meta['n_frames']} frames")
    server = ThreadingHTTPServer(
        (args.host, args.port), partial(Handler, trajectories=trajectories)
    )
    print(f"viewer at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
