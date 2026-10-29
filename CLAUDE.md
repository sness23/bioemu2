# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BioEmu2 (Microsoft Research AI for Science) is a diffusion-based foundation model for biomolecular structure and dynamics: structure prediction, equilibrium ensemble sampling, and time-lagged propagation (implicit transfer operator) for proteins, nucleic acids, ligands, and modified residues. Python package in `src/bioemu2`, CLI entry point `bioemu2` (`bioemu2.cli:main`). Runtime targets Linux + NVIDIA GPU; tests run on CPU in CI.

## Commands

Use `uv` (managed interpreters only — Triton needs Python headers; see `[tool.uv]` in `pyproject.toml`):

```bash
uv sync --extra cuda --extra dev          # GPU dev env (CC >= 7.5)
uv sync --extra cuda12 --extra dev        # V100 / CC 7.0; pass --extra cuda12 to EVERY `uv run` too,
                                          # or uv run reinstalls the default cuda build
uv sync --locked --extra dev              # what CI uses (CPU, no cuEquivariance)

uv run pytest tests                       # full suite
uv run pytest tests/denoising/test_edm.py::test_name -v   # single test
uv run pytest tests -m "not slow"         # markers: gpu, slow, serial, requires_4gpu
uv run pre-commit run --all-files         # ruff format + ruff check --fix + mypy + docs TOC
```

- CI (`.github/workflows/ci.yml`) runs pytest on Python 3.12/3.13/3.14 and pre-commit on 3.12 only.
- `requires_4gpu` tests auto-skip with <4 CUDA devices (`tests/conftest.py`); `gpu` tests skip via the `gpu_device` fixture.
- `tests/conftest.py` sets `OMP_NUM_THREADS=2` before importing torch so CPU float32 reductions are deterministic enough for tight tolerance tests — don't remove it.
- Offline PDB fixtures come from `tests/assets/rcsb_structures_*.tar.gz` (`pdb_files_dir` fixture). Model tests use `tests/assets/tiny_config.yaml` with randomized params, not real checkpoints.
- Ruff: line length 100, py312 target; `PLC0415` is enabled (no imports inside functions). Pre-commit's `check-added-large-files` caps files at 500 KB (except `uv.lock`).
- `scripts/generate_docs_toc.py` regenerates the `<!-- BEGIN/END GENERATED TOC -->` block in `docs/*.md` (long docs only); run via pre-commit after editing docs headings.
- Releases: `publish.yml` requires the release tag to match the static `version` in `pyproject.toml`.

## Architecture

Pipeline: **System → MSA → trunk embeddings → diffusion sampler → frames (`samples.xtc` + `topology.cif`)**.

- `data_interface/` — the `System` model (Protein/DNA/RNA/Ligand/Branched/ModifiedResidue/Contact), loading from PDB IDs, structure files, system YAML (`src/bioemu2/assets/examples/systems/`), and Boltz YAML conversion (`boltz.py`).
- `structure/` — builds all-atom inference inputs from a `System` (`preparation.py`, `inference_inputs.py`, polymer/CCD templates, batching).
- `msa/` — `MSA` objects, ColabFold retrieval, A3M loading, featurization.
- `biomoa/` — the neural network (`BioMoaCModel` in `models/score_model.py`): PairMixer trunk, diffusion module, and custom ops (triangle multiplication, pair-bias attention, Triton kernels). `shortcuts.py` provides stable import paths that Hydra configs in checkpoints `instantiate()` — renaming these classes breaks checkpoint loading.
- `model.py` — `load_model()` → `InferenceModel` protocol (`embed`, `compute_static_context`, `denoise`) wrapping the score model with precision/compile/embedding-cache handling. `checkpoints.py` resolves checkpoints from HF repo `microsoft/bioemu2`; published checkpoint names are immutable and listed in `SUPPORTED_CHECKPOINTS`.
- `conditioning/` — `SamplingMode` (structure vs. equilibrium vs. dynamics method conditioning) and contact conditioning.
- `denoising/` — EDM sampler (`edm.py`, `samplers.py`) and `steering/` (FKC/guided samplers, `Potential` definitions, geometry/RDKit constraints).
- `sampling.py` / `propagation.py` — functional API (`predict_structure`, `sample_equilibrium`, `propagate`, `simulate_trajectory`, `*_with_embeddings` variants). `inference.py` `EmuSession` wraps these and caches one embedding per session, fingerprinting system/contacts to detect mutation.
- `physicality.py` — post-sampling physical-validity checks and filtering.
- `context_parallel/` — a parallel reimplementation of the `biomoa` model/modules/ops on DTensors over an `n×n` GPU mesh (square number ≥4 GPUs). Launched via `torchrun --module bioemu2.cp ...` (`cp.py` → `cli.context_parallel_commands`). Changes to `biomoa` modules usually need matching changes here; tests in `tests/context_parallel/` check serial-vs-CP parity, often via `spawn_cpu_cp_test` in `helpers.py` (CPU ranks, no GPUs needed).
- `patches/` — runtime monkeypatches for mdtraj/OpenMM, imported first in `__init__.py`.

Public API is whatever `src/bioemu2/__init__.py` exports. Cache directories are controlled by `BIOEMU2_EMBEDS_CACHE`, `BIOEMU2_MSA_CACHE`, `BIOEMU2_CCD_CACHE`, `BIOEMU2_TEMPLATE_CACHE` (embedding caching is off unless set).

## Docs and examples

User docs live in `docs/*.md` (linked from `README.md`). `examples/*.py` are `# %%`-cell Python files (runnable as scripts or notebooks; generated `.ipynb` are gitignored) and need a GPU plus network access.
