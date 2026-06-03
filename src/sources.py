import torch # type: ignore
import numpy as np # type: ignore
from scipy.special import gammaln # type: ignore
from .simparams import SimParams
from .util import complex_to_real_dtype

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def gaussian_source(sim_params: SimParams, rsrc: float) -> torch.Tensor:
    # Gaussian source
    U_init = torch.zeros((sim_params.weights.shape[0], sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    R_sq = sim_params.X**2 + sim_params.Y**2
    U_init = torch.exp(-R_sq / (2 * rsrc**2)) * torch.exp(1j * 2*torch.pi * torch.rand(U_init.shape, dtype=torch.float32, device=sim_params.device))

    #zero out values below a certain threshold
    U_init[U_init.abs() < 1e-6] = 0

    return U_init

def plane_wave(sim_params: SimParams) -> torch.Tensor:
    U_init = torch.ones((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)

    return U_init

def half_plane_wave(sim_params: SimParams) -> torch.Tensor:
    U_init = torch.ones((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=sim_params.dtype, device=sim_params.device)
    U_init[:, :, :sim_params.Nx//2] = 0

    return U_init


def gsm_modes_1d(
    sim_params: SimParams,
    sigma_s: float,
    sigma_g: float,
    n_modes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hermite-Gaussian coherent modes for a 1D Gaussian Schell-model source.

    The GSM cross-spectral density is:
        W(x1, x2) = exp(-(x1^2+x2^2)/(4 sigma_s^2)) exp(-(x1-x2)^2/(2 sigma_g^2))

    Its Mercer decomposition yields modes that are Hermite-Gaussian functions
    with geometrically decaying eigenvalues.

    Args:
        sim_params: Provides the x coordinate grid, device, and dtype.
        sigma_s: RMS beam width (m). Controls the Gaussian intensity envelope.
                 Set large relative to the aperture for approximately uniform
                 illumination.
        sigma_g: RMS coherence length (m). Controls transverse spatial
                 coherence; larger values mean higher coherence and fewer
                 significant modes.
        n_modes: Number of coherent modes to compute.

    Returns:
        modes: Complex tensor of shape (n_modes, 1, Nx). Each slice [m, :, :]
               is the m-th coherent mode field.
        eigenvalues: Real tensor of shape (n_modes,). Incoherent weights for
                     each mode, normalized so that the truncated on-axis
                     spectral density equals 1.
    """
    real_dtype = complex_to_real_dtype(sim_params.dtype)
    x_np = sim_params.x.detach().cpu().numpy().astype(np.float64)
    Nx = len(x_np)

    a = 1.0 / (4.0 * sigma_s ** 2)
    b = 1.0 / (2.0 * sigma_g ** 2)
    c = np.sqrt(a * (a + 2.0 * b))

    denom = a + b + c
    beta_ratio = b / denom
    lambda_0 = np.sqrt(np.pi / denom)
    eigenvalues = np.array([lambda_0 * beta_ratio ** n for n in range(n_modes)])

    # Build modes via the stable three-term Hermite-Gaussian recurrence.
    z = x_np * np.sqrt(2.0 * c)
    gauss = np.exp(-c * x_np ** 2)
    prefactor = (2.0 * c / np.pi) ** 0.25

    modes = np.zeros((n_modes, Nx), dtype=np.float64)
    modes[0] = prefactor * gauss
    if n_modes > 1:
        modes[1] = np.sqrt(2.0) * z * modes[0]
    for n in range(1, n_modes - 1):
        modes[n + 1] = (
            np.sqrt(2.0 / (n + 1)) * z * modes[n]
            - np.sqrt(float(n) / (n + 1)) * modes[n - 1]
        )

    # Normalize so the truncated sum gives unit on-axis spectral density,
    # matching the plane-wave amplitude of 1.
    center = Nx // 2
    S0 = np.sum(eigenvalues * modes[:, center] ** 2)
    if S0 > 0:
        eigenvalues /= S0

    modes_t = torch.tensor(
        modes[:, np.newaxis, :], dtype=sim_params.dtype, device=sim_params.device
    )
    eigenvalues_t = torch.tensor(
        eigenvalues, dtype=real_dtype, device=sim_params.device
    )
    return modes_t, eigenvalues_t


def gsm_modes_2d_lg(
    sim_params: SimParams,
    sigma_s: float,
    sigma_g: float,
    n_modes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Laguerre-Gaussian coherent modes for the circularly symmetric 2D
    Gaussian Schell-model source.

    The separable Cartesian GSM used by ``gsm_modes_1d`` with identical
    sigma_s, sigma_g along x and y is equivalent to the circularly symmetric
    2D GSM

        W(r1, r2) = exp(-(|r1|^2 + |r2|^2) / (4 sigma_s^2))
                    * exp(-|r1 - r2|^2 / (2 sigma_g^2)),

    whose Mercer decomposition in polar coordinates is diagonalized by
    Laguerre-Gaussian modes

        LG_{p,l}(r, phi) = R_{p,|l|}(r) * exp(i l phi)

    with eigenvalues proportional to beta^(2p + |l|) for the same
    beta = b / (a + b + c) used in 1D. Through a rotation-invariant optical
    system only R_{p,|l|}(r) needs to be evolved, via an order-|l| Hankel
    transform.

    This function returns RADIAL profiles R_{p,|l|}(r) sampled on the uniform
    half-grid r = (0..N_rad-1) * dx with N_rad = Nx // 2. Along with each
    radial profile we return its azimuthal order m = |l| and a multiplicity
    (1 for m == 0, 2 for m >= 1, accounting for +l and -l modes having
    identical magnitude), so callers propagate each unique radial profile
    once and weight the accumulated intensity accordingly.

    Triangular truncation ``2p + m <= n_modes - 1`` matches the 1D tail
    tolerance exactly: the truncation error is O(beta^n_modes) in both bases.

    Normalization mirrors ``gsm_modes_1d``: after computing raw eigenvalues
    beta^(2p+m), all eigenvalues are rescaled so that the truncated on-axis
    spectral density

        S(0) = sum_{(p,m)} mult(m) * lambda_{p,m} * |R_{p,m}(0)|^2

    equals 1. Only m == 0 contributes at r == 0 because the r^m factor kills
    R_{p,m}(0) for m >= 1, so this directly matches the 1D convention of
    unit on-axis spectral density (plane-wave amplitude of 1).

    Args:
        sim_params: Provides Nx, dx, device, dtype.
        sigma_s: RMS beam width (m).
        sigma_g: RMS coherence length (m).
        n_modes: 1D truncation order reused as triangular-truncation order
                 K + 1 here (K = n_modes - 1).

    Returns:
        radials: Real tensor of shape (K_total, N_rad) holding R_{p,m}(r)
                 for all (p, m) with 2p + m <= n_modes - 1.
        m_orders: Int64 tensor of shape (K_total,) with m = |l| per mode.
        eigenvalues: Real tensor of shape (K_total,) with normalized
                     lambda_{p,m} (unit on-axis spectral density convention).
        multiplicities: Real tensor of shape (K_total,) with 1.0 for m == 0
                        and 2.0 for m >= 1.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")

    real_dtype = complex_to_real_dtype(sim_params.dtype)
    Nx = int(sim_params.Nx)
    dx = float(sim_params.dx)
    N_rad = Nx // 2
    if N_rad < 1:
        raise ValueError(f"Nx must be >= 2 for a radial half-grid, got Nx={Nx}")

    r_np = np.arange(N_rad, dtype=np.float64) * dx

    a = 1.0 / (4.0 * sigma_s ** 2)
    b = 1.0 / (2.0 * sigma_g ** 2)
    c = float(np.sqrt(a * (a + 2.0 * b)))
    denom = a + b + c
    beta = b / denom

    u = 2.0 * c * (r_np ** 2)
    exp_factor = np.exp(-c * r_np ** 2)
    r_scaled = r_np * np.sqrt(2.0 * c)

    K = int(n_modes - 1)

    radials_list: list[np.ndarray] = []
    m_orders_list: list[int] = []
    eigenvalues_list: list[float] = []
    multiplicities_list: list[float] = []

    for m in range(0, K + 1):
        p_max = (K - m) // 2
        if p_max < 0:
            continue

        # r^m factor in scaled units preserves R_{p,m}(0) == 0 exactly for m >= 1.
        if m == 0:
            r_pow_m = np.ones_like(r_scaled)
        else:
            r_pow_m = r_scaled ** m

        # Generalized Laguerre recurrence:
        # L_0^m = 1, and
        # (p+1) L_{p+1}^m = (2p + m + 1 - u) L_p^m - (p + m) L_{p-1}^m.
        L_prev = np.zeros_like(u)  # L_{-1} sentinel (annihilated by (p+m) factor at p=0)
        L_curr = np.ones_like(u)   # L_0^m

        for p in range(0, p_max + 1):
            L_p = L_curr

            # N_{p,m} = sqrt((2 c / pi) * p! / (p + m)!)
            log_N = 0.5 * (
                np.log(2.0 * c / np.pi)
                + gammaln(p + 1)
                - gammaln(p + m + 1)
            )
            N_pm = float(np.exp(log_N))

            R_pm = N_pm * r_pow_m * L_p * exp_factor

            radials_list.append(R_pm.astype(np.float64))
            m_orders_list.append(int(m))
            eigenvalues_list.append(float(beta ** (2 * p + m)))
            multiplicities_list.append(1.0 if m == 0 else 2.0)

            # Advance to p + 1 for next iteration.
            L_next = ((2.0 * p + m + 1.0 - u) * L_curr - (p + m) * L_prev) / float(p + 1)
            L_prev = L_curr
            L_curr = L_next

    radials_np = np.stack(radials_list, axis=0)
    m_orders_np = np.asarray(m_orders_list, dtype=np.int64)
    eigenvalues_np = np.asarray(eigenvalues_list, dtype=np.float64)
    multiplicities_np = np.asarray(multiplicities_list, dtype=np.float64)

    # Normalize to unit on-axis spectral density. Only m == 0 contributes at r = 0.
    R0_sq = radials_np[:, 0] ** 2
    S0 = float(np.sum(multiplicities_np * eigenvalues_np * R0_sq))
    if S0 > 0:
        eigenvalues_np = eigenvalues_np / S0

    radials_t = torch.tensor(radials_np, dtype=real_dtype, device=sim_params.device)
    m_orders_t = torch.tensor(m_orders_np, dtype=torch.int64, device=sim_params.device)
    eigenvalues_t = torch.tensor(eigenvalues_np, dtype=real_dtype, device=sim_params.device)
    multiplicities_t = torch.tensor(multiplicities_np, dtype=real_dtype, device=sim_params.device)

    return radials_t, m_orders_t, eigenvalues_t, multiplicities_t