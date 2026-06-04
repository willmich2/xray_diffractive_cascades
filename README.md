# Diffractive Cascades

Code accompanying the manuscript **"Diffractive cascades for polychromatic hard X-ray focusing"**.

This repository is organized as a reproducible pipeline with two main stages:

1. **Data generation (sweeps):** Run optimization sweeps that design diffractive cascades under various parameter configurations and save results to disk.
2. **Figure generation (notebooks):** Load the pre-computed optimization outputs and produce the manuscript figures.

Pre-generated data is included in `paper_data/`, so you can recreate all figures immediately using the notebooks without re-running any optimizations.

### Repository layout

- `src/` implements the forward models, optimization, and metrics.
- `paper/sweeps/` defines and runs the optimization sweeps that generate data.
- `paper/postprocess/` computes robustness metrics from sweep outputs.
- `paper/experiments/` runs one-off experiments (focal sweep comparison for depth of focus).
- `notebooks/` recreates manuscript figures from saved `.npz/.npy` data.
- `paper_data/` contains pre-generated optimization outputs consumed by the notebooks.
- `data/` contains tabulated optical constants (material refractive indices).
- `examples/` provides standalone scripts for quick single-optimization runs (useful for smoke tests or getting started).

### Naming convention

Notebooks and saved outputs are prefixed by the manuscript figure (and subfigure) they support, for example `fig2a_*` or `figA3b_*`. Sweep and experiment scripts write files of the form `{prefix}_params_{timestamp}.npy`, `{prefix}_sweep_arrays_{timestamp}.npy`, and `{prefix}_results_{timestamp}.npz` (or `{prefix}_results_{timestamp}_run_{i}.npz` for multi-run sweeps). The bundled files in `paper_data/` omit the timestamp suffix (e.g. `fig1_N_sweeps_params.npy`).

## Hardware Requirement

The optimization runs in this repo are computationally heavy. **Use a CUDA GPU** for practical runtimes.

- CPU-only runs are primarily useful for smoke tests.
- Sweeps and postprocessing scripts automatically use available GPUs.

## Installation

Use Python 3.11+ and install dependencies:

```bash
pip install -r requirements.txt
```

Set a data directory (recommended: `paper_data/`):

```bash
export DIFFRACTIVE_CASCADES_DATA_DIR=paper_data
```

## Recreating Figures from Existing Data

Open the notebook for the figure you want to reproduce and run all cells. Each notebook loads outputs from `paper_data/` using the matching `fig*` prefix.

| Figure | Notebook | Data prefix(es) in `paper_data/` |
|--------|----------|----------------------------------|
| Fig. 1(c) | `notebooks/fig1c_element_visualization.ipynb` | `fig1_N_sweeps` |
| Fig. 1(d) | `notebooks/fig1d_focal_spot_comparison.ipynb` | `fig1d_xray_focusing_testing` |
| Fig. 1(e) | `notebooks/fig1e_Nelem_sweep.ipynb` | `fig1_N_sweeps` |
| Fig. 2(a) | `notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb` | `fig2a_bandwidth_energy_sweep`, `fig2a_thickness_energy_sweep` |
| Fig. 2(b) | `notebooks/fig2b_mfs_sweep.ipynb` | `fig2b_Nelem_min_feature_sweep` |
| Fig. 2(c) | `notebooks/fig2c_aspect_ratio_scaling.ipynb` | `fig2c_thickness_energy_sweep` |
| Fig. 3(d) | `notebooks/fig3d_robustness.ipynb` | `fig3d_placement_robustness_results`, `fig3d_erosion_dilation_robustness_results`, `fig3d_thermal_robustness_results` |
| Fig. 4(b) | `notebooks/fig4b_depth_of_focus.ipynb` | `fig4b_xray_focusing_focal_sweep_comparison` |
| Fig. A.1 | `notebooks/figA1_partial_coherence.ipynb` | `figA1_coherence_illumination_sweep` |
| Fig. A.3(a) | `notebooks/figA3a_focal_length.ipynb` | `figA3a_focal_length_sweeps` |
| Fig. A.3(b) | `notebooks/figA3b_inter_elem_dist.ipynb` | `figA3b_inter_elem_dist_sweeps` |

Fig. 3(d) also uses sidewall data for Fig. A.4(b): `figA4b_sidewall_robustness_results`.

## Re-running Optimizations (Data Generation)

To regenerate the optimization data from scratch (requires a CUDA GPU for practical runtimes), use the paper entrypoint:

```bash
python paper/reproduce.py list
```

```bash
python paper/reproduce.py run <target_key>
```

You can override runtime settings:

```bash
python paper/reproduce.py run fig1e --data-dir paper_data --workers-per-gpu 2 --max-workers 8
```

### Main-text targets

- `fig1e` → `notebooks/fig1e_Nelem_sweep.ipynb`, data prefix `fig1_N_sweeps` (study `n_sweeps`).
- `fig1c` → `notebooks/fig1c_element_visualization.ipynb`, same `fig1_N_sweeps` run as `fig1e`.
- `fig2a_bandwidth` → `notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb`, prefix `fig2a_bandwidth_energy_sweep` (study `bandwidth_energy`).
- `fig2a_thickness` → same notebook, prefix `fig2a_thickness_energy_sweep` (study `thickness_energy_fig2a`).
- `fig2b` → `notebooks/fig2b_mfs_sweep.ipynb`, prefix `fig2b_Nelem_min_feature_sweep` (study `nelem_min_feature`).
- `fig2c` → `notebooks/fig2c_aspect_ratio_scaling.ipynb`, prefix `fig2c_thickness_energy_sweep` (study `thickness_energy_main`).
- `fig3_placement`, `fig3_erosion_dilation`, `fig3_thermal` → `notebooks/fig3d_robustness.ipynb`; postprocess outputs `fig3d_*_robustness_results`.
- `figA4_sidewall` → same robustness notebook; postprocess output `figA4b_sidewall_robustness_results`.
- `fig4b` → `notebooks/fig4b_depth_of_focus.ipynb`, prefix `fig4b_xray_focusing_focal_sweep_comparison` (script `paper/experiments/xray_focusing_focal_sweep_comparison.py`).

`notebooks/fig1d_focal_spot_comparison.ipynb` (Fig. 1(d)) uses `fig1d_xray_focusing_testing_*` from `examples/xray_focusing_testing.py` and can be run from the included `paper_data/` without a separate `reproduce.py` target.

### Appendix targets

- `figA1` → `notebooks/figA1_partial_coherence.ipynb`, prefix `figA1_coherence_illumination_sweep` (study `coherence_illumination`).
- `figA3_focal` → `notebooks/figA3a_focal_length.ipynb`, prefix `figA3a_focal_length_sweeps` (study `focal_length`).
- `figA3_inter` → `notebooks/figA3b_inter_elem_dist.ipynb`, prefix `figA3b_inter_elem_dist_sweeps` (study `inter_element_distance`).

After re-running a sweep or experiment, update the timestamp `ID` in the corresponding notebook if it no longer matches the new `fig*_*_{timestamp}` filenames on disk.

## Examples

The `examples/` directory contains standalone scripts that run a single optimization:

```bash
python examples/xray_focusing_testing.py
python examples/xray_focusing_testing_partial_coherence.py
```

`xray_focusing_testing.py` writes `fig1d_xray_focusing_testing_params_{timestamp}.npy` and `fig1d_xray_focusing_testing_results_{timestamp}.npz` for `fig1d_focal_spot_comparison.ipynb`.

These scripts are useful for verifying your installation or experimenting with parameters outside of the full sweep pipeline. On a GPU, each completes in a few minutes; a SLURM launcher is available at `hpc/slurm/run_example_gpu.sh`.

### Console output

Runs print standardized, timestamped lines via `src/console.py`, for example:

```text
[14:32:01] [sweep] INFO: parallel pool: 2 GPU(s), 2 worker(s)/GPU, 4 worker(s) total
[14:32:01] [optimizer] INFO: beta stage 1/6: beta=1, constraints=off
```

Set `DIFFRACTIVE_CASCADES_QUIET=1` to suppress these messages.

## Direct Sweep Runner

If you want to run individual optimization sweeps directly (bypassing `reproduce.py`):

```bash
python paper/sweeps/run_sweep.py --list-studies
python paper/sweeps/run_sweep.py --study n_sweeps --save-dir paper_data
```

Study keys (`n_sweeps`, `bandwidth_energy`, `thickness_energy_fig2a`, etc.) map to the `fig*` save prefixes above. Each sweep runs a batch of optimizations over a parameter grid and saves the results with notebook-compatible axis naming and output schema. Projected element densities (`opt_rhos`) are stored as compressed `bool` arrays (binary solid/void); legacy `float64` files are still accepted by notebooks and postprocessing.

## Robustness Postprocessing Notes

Robustness scripts read a completed `fig1_N_sweeps` run and write figure-prefixed results:

| Script | Output prefix |
|--------|----------------|
| `paper/postprocess/placement_robustness.py` | `fig3d_placement_robustness_results` |
| `paper/postprocess/erosion_dilation_robustness.py` | `fig3d_erosion_dilation_robustness_results` |
| `paper/postprocess/thermal_robustness.py` | `fig3d_thermal_robustness_results` |
| `paper/postprocess/sidewall_robustness.py` | `figA4b_sidewall_robustness_results` |

Defaults are set to the manuscript `fig1_N_sweeps` run ID, but you can pass your own:

```bash
python paper/postprocess/placement_robustness.py --base-id <FIG1_N_SWEEP_ID> --run-id 0 --data-dir paper_data
```

## SLURM

Cluster launchers are in `hpc/slurm/`:

```bash
sbatch hpc/slurm/run_sweep.sh n_sweeps --save-dir paper_data
```

## Citation

If you use this code, please cite:

William Michaels, Simo Pajovic, Joshua Chen, Charles Roques-Carmes, and Marin Soljačić, "Diffractive cascades for polychromatic hard X-ray focusing," arXiv:2605.15526, 2026.
