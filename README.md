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

If you just want to reproduce the manuscript figures using the included pre-generated data, open any notebook in `notebooks/` and run it. Each notebook loads optimization outputs from `paper_data/` and produces the corresponding figure.

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

- `fig1e` (`notebooks/fig1e_Nelem_sweep.ipynb`): material/depth scaling (`fig1_N_sweeps`).
- `fig1c` (`notebooks/fig1c_element_visualization.ipynb`): element visualizations from the same `fig1_N_sweeps` run.
- `fig2a_bandwidth` (`notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb`): bandwidth-energy sweep.
- `fig2a_thickness` (`notebooks/fig2a_energy_bandwidth_aspect_ratio.ipynb`): thickness-energy sweep on the 30x30 Fig. 2(a) grid.
- `fig2b` (`notebooks/fig2b_mfs_sweep.ipynb`): minimum-feature-size tradeoff sweep.
- `fig2c` (`notebooks/fig2c_aspect_ratio_scaling.ipynb`): thickness-energy sweep for aspect-ratio scaling.
- `fig3_placement`, `fig3_erosion_dilation`, `fig3_thermal`, `figA4_sidewall` (`notebooks/fig3d_robustness.ipynb`): robustness postprocessing.
- `fig4b` (`notebooks/fig4b_depth_of_focus.ipynb`): depth-of-focus experiment.

`notebooks/fig1d_focal_spot_comparison.ipynb` (Fig. 1(d) focal spot profile) uses data from `examples/xray_focusing_testing.py` and can be run directly from the included `paper_data/` without a separate reproduce target.

### Appendix targets

- `figA1` (`notebooks/figA1_partial_coherence.ipynb`): partial coherence sweep.
- `figA3_focal` (`notebooks/figA3a_focal_length.ipynb`): focal-length sweep.
- `figA3_inter` (`notebooks/figA3b_inter_elem_dist.ipynb`): inter-element-distance sweep.

Notebook cells near the top define dataset IDs (timestamp strings). The defaults point to the included pre-generated data. If you re-run an optimization sweep, update the `ID` variable in the corresponding notebook to the new timestamp so it loads your fresh outputs.

## Examples

The `examples/` directory contains standalone scripts that run a single optimization:

```bash
python examples/xray_focusing_testing.py
python examples/xray_focusing_testing_partial_coherence.py
```

These are useful for verifying your installation or experimenting with parameters outside of the full sweep pipeline. On a GPU, each completes in a few minutes; a SLURM launcher is available at `hpc/slurm/run_example_gpu.sh`.

## Direct Sweep Runner

If you want to run individual optimization sweeps directly (bypassing `reproduce.py`):

```bash
python paper/sweeps/run_sweep.py --list-studies
python paper/sweeps/run_sweep.py --study n_sweeps --save-dir paper_data
```

Each sweep runs a batch of optimizations over a parameter grid and saves the results. The sweep configs are organized for notebook compatibility (axis naming and output schema).

## Robustness Postprocessing Notes

Robustness scripts consume an existing `fig1_N_sweeps` run ID. Defaults are set to the manuscript ID, but you can pass your own:

```bash
python paper/postprocess/placement_robustness.py --base-id <FIG1_N_SWEEP_ID> --run-id 0 --data-dir paper_data
```

Equivalent arguments exist for:

- `paper/postprocess/erosion_dilation_robustness.py`
- `paper/postprocess/thermal_robustness.py`
- `paper/postprocess/sidewall_robustness.py`

## SLURM

Cluster launchers are in `hpc/slurm/`:

```bash
sbatch hpc/slurm/run_sweep.sh n_sweeps --save-dir paper_data
```

## Citation

If you use this code, please cite:

William Michaels, Simo Pajovic, Joshua Chen, Charles Roques-Carmes, and Marin Soljačić, "Diffractive cascades for polychromatic hard X-ray focusing," arXiv:2605.15526, 2026.
