from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SweepStudy:
    key: str
    config_module: str
    manuscript_result: str
    description: str
    notebooks: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    requires_gpu: bool = True


PAPER_SWEEP_STUDIES: tuple[SweepStudy, ...] = (
    SweepStudy(
        key="n_sweeps",
        config_module="paper.sweeps.configs.n_sweeps",
        manuscript_result="Fig. 1(e) and Fig. 1(c) element visualizations",
        description="Material and cascade-depth scaling",
        notebooks=("notebooks/Nelem_sweep.ipynb", "notebooks/element_visualization.ipynb"),
        aliases=("n", "nelem"),
    ),
    SweepStudy(
        key="bandwidth_energy",
        config_module="paper.sweeps.configs.bandwidth_energy_sweep",
        manuscript_result="Fig. 2(a) bandwidth-energy panel",
        description="Bandwidth vs energy map",
        notebooks=("notebooks/energy_bw_ar.ipynb",),
        aliases=("bandwidth", "bw"),
    ),
    SweepStudy(
        key="thickness_energy_main",
        config_module="paper.sweeps.configs.thickness_energy_sweep",
        manuscript_result="Fig. 2(c) aspect-ratio scaling",
        description="Thickness-energy sweep used for aspect-ratio scaling",
        notebooks=("notebooks/aspect_ratio_scaling.ipynb",),
        aliases=("thickness", "aspect_ratio"),
    ),
    SweepStudy(
        key="thickness_energy_fig2a",
        config_module="paper.sweeps.configs.thickness_energy_sweep_fig2a",
        manuscript_result="Fig. 2(a) aspect-ratio/energy panel",
        description="Thickness-energy sweep on 30x30 grid for Fig. 2(a)",
        notebooks=("notebooks/energy_bw_ar.ipynb",),
        aliases=("thickness_energy_dense", "thickness_fig2a"),
    ),
    SweepStudy(
        key="nelem_min_feature",
        config_module="paper.sweeps.configs.nelem_min_feature_sweep",
        manuscript_result="Fig. 2(b)",
        description="Spot-size vs efficiency tradeoff",
        notebooks=("notebooks/mfs_sweep.ipynb",),
        aliases=("mfs", "min_feature"),
    ),
    SweepStudy(
        key="coherence_illumination",
        config_module="paper.sweeps.configs.coherence_illumination_sweep",
        manuscript_result="Appendix B, Fig. A.1",
        description="Partial spatial coherence sweep",
        notebooks=("notebooks/partial_coherence.ipynb",),
        aliases=("coherence",),
    ),
    SweepStudy(
        key="focal_length",
        config_module="paper.sweeps.configs.focal_length_sweeps",
        manuscript_result="Appendix D, Fig. A.3(a)",
        description="Focal-length scaling",
        notebooks=("notebooks/focal_length.ipynb",),
        aliases=("focal",),
    ),
    SweepStudy(
        key="inter_element_distance",
        config_module="paper.sweeps.configs.inter_elem_dist_sweeps",
        manuscript_result="Appendix D, Fig. A.3(b)",
        description="Inter-element spacing scaling",
        notebooks=("notebooks/inter_elem_dist.ipynb",),
        aliases=("inter_elem_dist", "inter_elem"),
    ),
)


_STUDY_BY_KEY = {study.key: study for study in PAPER_SWEEP_STUDIES}
_STUDY_BY_ALIAS = {
    alias: study for study in PAPER_SWEEP_STUDIES for alias in (study.key, *study.aliases)
}


def iter_studies() -> tuple[SweepStudy, ...]:
    return PAPER_SWEEP_STUDIES


def resolve_study(study_name: str) -> SweepStudy:
    key = study_name.strip()
    if key not in _STUDY_BY_ALIAS:
        valid = ", ".join(sorted(_STUDY_BY_ALIAS.keys()))
        raise KeyError(f"Unknown sweep study '{study_name}'. Valid studies/aliases: {valid}")
    return _STUDY_BY_ALIAS[key]
