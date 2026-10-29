# BioEmu2 flythrough viewer

A standalone browser page for flying through a BioEmu2 trajectory while it plays,
in the spirit of Descent 2: keyboard and mouse, six degrees of freedom, drift and
roll. Atoms are drawn old-school as line bonds, each half of a bond coloured by
its own atom. The viewer only shows the trajectory for now. Later, the plan is to
let you push on atoms and have that feed back into the simulation (see
[Roadmap](#roadmap-pushing-on-atoms)).

## Quick start

```bash
uv sync --extra cuda

# 1. Simulate a trajectory (one equilibrium sample, then N propagation steps).
uv run python viewer/make_trajectory.py --pdb-id 1UBQ --steps 100 --lag-time-ns 1 \
    --out outputs/fly/ubiquitin

# 2. Serve one or more trajectory directories, then open http://127.0.0.1:8765/
uv run python viewer/server.py outputs/fly/ubiquitin outputs/fly/chignolin
```

`make_trajectory.py` takes `--pdb-id` or `--system <yaml>`, plus `--steps`,
`--lag-time-ns`, `--seed`, and `--device cpu` (runs in float32 when the GPU is busy
or too small). Any directory holding `topology.cif` + `samples.xtc` from
`save_frames` can be served.

## Controls

Click the page to grab the mouse; Esc releases it.

| Input | Action |
|-------|--------|
| mouse, arrow keys | pitch / yaw |
| Q / E | roll |
| W / S | thrust forward / back |
| A / D | slide left / right |
| R / F | slide up / down |
| Shift | boost (×3 thrust) |
| Space | play / pause |
| , / . | step one frame back / forward (pauses) |
| [ / ] | halve / double playback rate (0.25–64 frames/s, starts at 4) |
| - / = | pull the depth fade closer / push it further |
| T | next trajectory |
| C | recentre the camera |
| H | hide / show the key list |

The HUD shows a crosshair, trajectory name, simulation time (ns), frame, play
state and rate, and ship speed in Å/s.

## How it works

```
make_trajectory.py ──► outputs/<name>/{topology.cif, samples.xtc}
                                   │
server.py (stdlib HTTP) ◄──────────┘   loads with mdtraj, superposes on frame 0 (CA),
   │                                   centres, converts nm → Å, derives bonds
   ├── GET /                     index.html
   ├── GET /api/list             ["chignolin", ...]
   ├── GET /api/traj/<i>/meta    elements, bonds, times_ps, n_atoms, n_frames
   └── GET /api/traj/<i>/coords  float32 [frames × atoms × 3], little-endian, Å
                                   │
index.html (three.js 0.160 from jsDelivr) ◄┘
```

**Bonds.** Bonds come from the CIF topology. If it has none, `create_standard_bonds()`
fills them in, and any atom still unbonded (ligands, odd residues) gets bonds to
neighbours within 1.9 Å in frame 0. Bonds are fixed for the whole trajectory.

**Half-bond colouring.** Each bond is two line segments, atom → midpoint → atom,
and each segment takes its own atom's Jmol CPK colour (C grey, N blue, O red,
S yellow, P orange, …; unknown elements are magenta). It's all one
`THREE.LineSegments` with vertex colours, rewritten every frame.

**Depth fade.** Linear `THREE.Fog` to black. It starts at 0.5× and ends at 4× the
molecule's radius; `-` and `=` scale the far edge.

**Smooth playback.** BioEmu2 steps are at least 1 ns apart (the model supports
1–1000 ns lags), about a thousand times coarser than MD output. Played raw, side
chains would snap between poses. The page blends between frames with a
Catmull-Rom spline through the four nearest frames, so motion looks continuous,
like MD. **This is cosmetic only:** the in-between positions aren't physical,
and atoms can pass through each other between real frames.

**Flight model** (constants at the top of the script in `index.html`):
- Thrust is applied along the camera's local axes, then velocity decays
  exponentially (`DRAG = 0.6`/s). Thrust scales with the molecule
  (`THRUST = 0.8` radii/s²), so a 10 Å peptide and a 60 Å protein take about the
  same time to cross. Top speed is about 1.3 radii/s, ×3 with boost. After letting
  go you coast for a second or two.
- Turning is rate-based. Mouse movement and arrow keys add angular velocity, and
  Q/E add roll, all damped by `TURN_DAMPING = 8`/s.
- Rotation is applied in the camera's local frame as quaternions, so there's no
  gimbal lock or fixed "up", as in Descent.

## Status (2026-10-27)

**Done**
- `server.py`, `index.html` and `make_trajectory.py`, all in this directory.
- Tested in headless Chromium on a jittered copy of PDB 7WOI (3,176 atoms, 20
  frames) and on real chignolin output:
  - endpoints work, the page renders, the HUD updates
  - thrust works, coasting decays, roll works, pause works
  - fixed thrust (20 Å/s²) shot straight through chignolin, so thrust now scales
    with molecule size
  - half-bond colours are correct
- Not tested: mouse steering. The headless browser can't grab the mouse, so that
  still needs a hands-on try.
- **Chignolin** (`outputs/fly/chignolin`): 10 residues, 76 heavy atoms, 101 frames
  at 1 ns = 100 ns. It ran on the CPU in float32 in 73 minutes (about 44 s per
  step).

**On the GPU (2026-10-28).** The RTX 2070 (8 GB) was shared with the
`planetar_fusion` training run (about 2.7 GB) and Xorg (about 2.8 GB), and the first
GPU attempt ran out of memory loading the weights. A background job waited for 4.5
GB of free GPU memory, which came free when training finished. It runs these at
about 15 s per step, three times faster than the CPU:
- **Ubiquitin** (1UBQ): 100 steps × 1 ns, written to `outputs/fly/ubiquitin`.
- **Adenylate kinase** (1AKE): 100 steps × 10 ns, written to `outputs/fly/adk`,
  chosen because its lid domain swings open and shut.

Logs go to `outputs-ubiquitin.log` and `outputs-adk.log` in the repo root.
- Ubiquitin finished: 601 heavy atoms, 101 frames, about 26 minutes.
- ADK ran out of GPU memory before sampling. It needs about 4.5 GB and about 4.3
  GB was free, and then a new training run took the GPU. Retry it with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` once the GPU is free.

## Known limitations

- The trajectory loops, so it jumps from the last frame back to frame 1.
- BioEmu2 outputs heavy atoms only, so there are no hydrogens.
- The blend between frames is visual only (see above).
- `-` can pull the fade's far edge in front of its near edge, which looks odd but
  is harmless.
- One trajectory per directory. `make_trajectory.py` simulates one trajectory
  from one equilibrium sample.

## Roadmap: pushing on atoms

The goal is to grab or push atoms while flying and have the push become part of
the simulation. BioEmu2 has no force-integration step: each propagation step
draws a new structure 1 ns or more ahead. A per-femtosecond push has nothing to
attach to. Two routes:

1. **Steering (BioEmu2 only).** Turn the push into a steering potential
   (`bioemu2.denoising.steering.Potential`, see `docs/steering.md` and
   `docs/potentials.md`) that biases the next propagation step, for example a
   harmonic pull of atom *i* toward a target point. It reacts coarsely, one step at
   a time, but stays within BioEmu2.
2. **Hybrid with OpenMM.** OpenMM is already a dependency. Run real MD for fine
   motion and take the user's forces every timestep as a custom external force,
   with BioEmu2 for big conformational jumps. This gives real MD motion with
   forces, but needs a force field and solvent setup.

Either way, `server.py` is the place to plug it in. It already sits between the
simulation and the page. It would gain a WebSocket (or a POST endpoint) that
takes `{atom, force}` from the page and streams new frames back, instead of
serving a fixed file.

## Files

| File | Purpose |
|------|---------|
| `index.html` | the page: rendering, flight model, HUD, playback |
| `server.py` | loads trajectories and serves the page plus the JSON/binary API |
| `make_trajectory.py` | simulates one BioEmu2 trajectory and saves it with `save_frames` |
