import torch # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass 
from .simparams import SimParams
from .util import refractive_index_at_wvl, complex_to_real_dtype
from .propagation import angular_spectrum_propagation

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def _shift_transmission(
    transmission: torch.Tensor,
    center: tuple[float, float],
    dx: float,
) -> torch.Tensor:
    """
    Shift a 2D or 3D transmission tensor by a physical center offset.

    - For 3D tensors, the expected shape is (C, H, W).
    - For 2D tensors, the expected shape is (H, W).

    The original transmission is assumed to be centered at (0, 0) in the
    simulation coordinates. A non-zero `center` translates the transmission
    so that its center lies at (center_x, center_y). Regions shifted out of
    the domain are discarded, and regions shifted in are filled with unity
    transmission (i.e., "blank" values).
    """
    if center == (0.0, 0.0):
        return transmission

    center_x, center_y = center
    # Convert physical shift (meters) to integer pixel shifts.
    shift_x = int(round(center_x / dx))
    shift_y = int(round(center_y / dx))

    # Handle both 2D (H, W) and 3D (C, H, W) tensors.
    if transmission.ndim == 3:
        _, H, W = transmission.shape
        prefix_slices = [slice(None)]
    elif transmission.ndim == 2:
        H, W = transmission.shape
        prefix_slices = []
    else:
        # For unexpected shapes, just return the input unchanged.
        return transmission

    shifted = torch.ones_like(transmission)

    # y-direction indices
    if shift_y >= 0:
        src_y0, src_y1 = 0, max(0, H - shift_y)
        dst_y0, dst_y1 = shift_y, shift_y + (src_y1 - src_y0)
    else:
        src_y0, src_y1 = -shift_y, H
        dst_y0, dst_y1 = 0, H + shift_y

    # x-direction indices
    if shift_x >= 0:
        src_x0, src_x1 = 0, max(0, W - shift_x)
        dst_x0, dst_x1 = shift_x, shift_x + (src_x1 - src_x0)
    else:
        src_x0, src_x1 = -shift_x, W
        dst_x0, dst_x1 = 0, W + shift_x

    # Only perform the copy if there is a valid overlap.
    if src_y1 > src_y0 and src_x1 > src_x0 and dst_y1 > dst_y0 and dst_x1 > dst_x0:
        src_idx = tuple(prefix_slices + [slice(src_y0, src_y1), slice(src_x0, src_x1)])
        dst_idx = tuple(prefix_slices + [slice(dst_y0, dst_y1), slice(dst_x0, dst_x1)])
        shifted[dst_idx] = transmission[src_idx]

    return shifted


@dataclass
class ArbitraryElement:
    """
    An element with arbitrary, spatially varying refractive index.
    Includes original and new batched methods for applying the element.
    """
    name: str
    thickness: float
    elem_map: list[float]
    gap_map: list[float]
    x: torch.Tensor
    membrane_thickness: float = 0.0
    membrane_map: list[float] = None
    center: tuple[float, float] = (0.0, 0.0)

    def __str__(self):
        return (
            f"ArbitraryElement(name={self.name}, thickness={self.thickness}, "
            f"elem_map={self.elem_map}, gap_map={self.gap_map}, x={self.x}, "
            f"membrane_thickness={self.membrane_thickness}, membrane_map={self.membrane_map}, "
            f"center={self.center})"
        )

    def __copy__(self):
        return ArbitraryElement(
            name=self.name, 
            thickness=self.thickness, 
            elem_map=self.elem_map, 
            gap_map=self.gap_map, 
            x=self.x,
            membrane_thickness=self.membrane_thickness,
            membrane_map=self.membrane_map,
            center=self.center,
            )

    def transmission(self, lams_tensor: torch.Tensor, sim_params: SimParams) -> torch.Tensor:
        """
        Calculates the transmission map for a batch of wavelengths simultaneously.

        Args:
            lams_tensor (torch.Tensor): A 1D tensor of wavelengths.
            sim_params (SimParams): Simulation parameters.

        Returns:
            torch.Tensor: A 3D tensor of transmission maps of shape (num_wavelengths, Ny, Nx).
        """
        
        x_tensor = self.x.to(sim_params.device)
        # x_tensor could be smaller than (Ny, Nx), so we need to pad it with zeros
        pad_left = (sim_params.Nx - x_tensor.shape[1]) // 2
        pad_right = sim_params.Nx - x_tensor.shape[1] - pad_left
        pad_top = (sim_params.Ny - x_tensor.shape[0]) // 2
        pad_bottom = sim_params.Ny - x_tensor.shape[0] - pad_top
        x_tensor = torch.nn.functional.pad(x_tensor, (pad_left, pad_right, pad_top, pad_bottom))

        # Calculate refractive indices for all wavelengths at once.
        # n_elem and n_gap will be 1D tensors of shape (num_wavelengths,).
        n_elem = refractive_index_at_wvl(lams_tensor, self.elem_map)
        n_gap = refractive_index_at_wvl(lams_tensor, self.gap_map)

        # Reshape n_elem and n_gap to (C, 1, 1) to enable broadcasting
        # with x_tensor, which has shape (H, W).
        n_elem_b = n_elem.view(-1, 1, 1)
        n_gap_b = n_gap.view(-1, 1, 1)

        # n_eff will have shape (C, H, W) after broadcasting.
        n_eff = n_elem_b * x_tensor + n_gap_b * (1 - x_tensor)
        del x_tensor

        # k0 (wave number) will be a 1D tensor of shape (C,).
        k0 = 2 * torch.pi / lams_tensor

        # Reshape k0 to (C, 1, 1) for broadcasting and calculate the complex phase.
        phase = k0.view(-1, 1, 1) * (n_eff - 1) * self.thickness
        del n_eff
        # Calculate element transmission
        element_transmission = torch.exp(1j * phase)
        del phase
        # Apply membrane transmission if membrane is present
        if self.membrane_thickness > 0 and self.membrane_map is not None:
            n_membrane = refractive_index_at_wvl(lams_tensor, self.membrane_map)
            n_membrane_b = n_membrane.view(-1, 1, 1)
            membrane_phase = k0.view(-1, 1, 1) * (n_membrane_b - 1) * self.membrane_thickness
            membrane_transmission = torch.exp(1j * membrane_phase)
            # Membrane is applied before the element
            element_transmission = membrane_transmission * element_transmission

        # Return the complex transmission map of shape (C, H, W), shifted so
        # that its center lies at `self.center`.
        return _shift_transmission(element_transmission, self.center, sim_params.dx)

    def apply_element(self, U: torch.Tensor, sim_params: SimParams) -> torch.Tensor:
        """
        Applies the element to a batch of fields using a single vectorized operation.

        Args:
            U (torch.Tensor): The input tensor of shape (num_wavelengths, Ny, Nx).
            sim_params (SimParams): Simulation parameters.

        Returns:
            torch.Tensor: The output tensor after applying the element.
        """
        complex_dtype = sim_params.dtype
        real_dtype = complex_to_real_dtype(complex_dtype)
        # Ensure wavelengths are a float tensor on the correct device.
        lams_tensor = torch.as_tensor(sim_params.lams, dtype=real_dtype, device=sim_params.device)

        # Get the transmission for all wavelengths at once.
        # The result `transmission` will have shape (C, Ny, Nx).
        transmission = self.transmission(lams_tensor, sim_params)

        # Apply the transmission to the input tensor U. This is an element-wise
        # multiplication of two (C, Ny, Nx) tensors.
        # Ensure dtypes match for the multiplication.
        U_f = U * transmission.to(U.dtype)
        
        return U_f

    def apply_element_sliced(self, U: torch.Tensor, slice_thickness: float, sim_params: SimParams):
        U_f = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=U.dtype, device=sim_params.device)
        
        for i, lam in enumerate(sim_params.lams):
            t = self.thickness
            n_slices = int(t // slice_thickness)
            for j in range(n_slices):
                t_slice = slice_thickness
                if j == n_slices - 1:
                    t_slice = t - j * slice_thickness
                slice_element = self.__copy__()
                slice_element.thickness = t_slice

                transmission = slice_element.transmission(lam, sim_params)
                U_lam = U[i, :, :] * transmission
                U_lam = angular_spectrum_propagation(U_lam, lam, t_slice, sim_params.dx, sim_params.device)
                U_f[i, :, :] = U_lam
        return U_f

@dataclass
class ZonePlate:
    name: str
    thickness: float
    min_feature_size: float
    elem_map: list[np.ndarray]
    gap_map: list[np.ndarray]
    f: float
    membrane_thickness: float = 0.0
    membrane_map: list[np.ndarray] = None
    center: tuple[float, float] = (0.0, 0.0)

    def __str__(self):
        return (
            f"ZonePlate(name={self.name}, thickness={self.thickness}, "
            f"min_feature_size={self.min_feature_size}, elem_map={self.elem_map}, "
            f"gap_map={self.gap_map}, membrane_thickness={self.membrane_thickness}, "
            f"membrane_map={self.membrane_map}, center={self.center})"
        )

    def __copy__(self):
        return ZonePlate(
            name=self.name, 
            thickness=self.thickness, 
            min_feature_size=self.min_feature_size,
            elem_map=self.elem_map,
            gap_map=self.gap_map,
            f=self.f,
            membrane_thickness=self.membrane_thickness,
            membrane_map=self.membrane_map,
            center=self.center,
            )
    
    def transmission(self, lam_inc: torch.Tensor, lam_des: torch.Tensor, sim_params: SimParams):
        """
        Calculates the transmission function of the zone plate.
        
        The zone plate pattern is generated up to a maximum radius determined by
        the minimum feature size. Beyond this radius, the transmission is
        that of the gap material.
        """
        complex_dtype = sim_params.dtype
        real_dtype = complex_to_real_dtype(complex_dtype)
        pi = torch.acos(torch.tensor(-1.0, dtype=real_dtype, device=sim_params.device))
        
        n_elem = refractive_index_at_wvl(lam_inc, self.elem_map)
        n_gap = refractive_index_at_wvl(lam_inc, self.gap_map)
        
        # Calculate radial distance from center for all points in the grid
        R_squared = sim_params.X**2 + sim_params.Y**2
        R = torch.sqrt(R_squared)

        R_cutoff = (lam_des * self.f) / (2 * self.min_feature_size)
        
        # Calculate the path difference to determine the zone number for each point
        # path_diff = torch.sqrt(R_squared + self.f**2) - self.f
        # zone_number = torch.floor(path_diff / (lam / 2.0))

        zone_number = torch.floor(R_squared / (lam_des * self.f))

        # Define the complex transmission for the element and gap materials
        trans_elem = torch.exp(1j * 2 * pi * (n_elem - 1) * self.thickness / lam_inc)
        trans_gap = torch.exp(1j * 2 * pi * (n_gap - 1) * self.thickness / lam_inc)

        # Create the ideal, infinite zone plate pattern based on the zone number
        # Even zones get the gap transmission, odd zones get the element transmission.
        zp_pattern = torch.where(zone_number % 2 == 0, 
                                  trans_gap,
                                  trans_elem)
        
        transmission = torch.where(R <= R_cutoff,
                                   zp_pattern,
                                   trans_elem)

        # Apply membrane transmission if membrane is present
        if self.membrane_thickness > 0 and self.membrane_map is not None:
            n_membrane = refractive_index_at_wvl(lam_inc, self.membrane_map)
            membrane_trans = torch.exp(1j * 2 * pi * (n_membrane - 1) * self.membrane_thickness / lam_inc)
            # Membrane is uniform across the entire domain and applied before the element
            membrane_transmission = torch.ones_like(transmission) * membrane_trans
            transmission = membrane_transmission * transmission

        # Return the complex transmission map shifted so that its center lies
        # at `self.center`.
        return _shift_transmission(transmission, self.center, sim_params.dx)

    def apply_element(self, U: torch.Tensor, sim_params: SimParams):
        U_f = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=U.dtype, device=sim_params.device)
        # zone plate profile changes with wavelength, so we need to use the maximum wavelength to maintain a constant profile
        max_lam = sim_params.lams[torch.argmax(sim_params.weights)]
        for i, lam in enumerate(sim_params.lams):
            transmission = self.transmission(lam, max_lam, sim_params)
            U_lam = U[i, :, :] * transmission
            U_f[i, :, :] = U_lam
        return U_f

    def apply_element_sliced(self, U: torch.Tensor, slice_thickness: float, sim_params: SimParams):
        U_f = torch.zeros((len(sim_params.weights), sim_params.Ny, sim_params.Nx), dtype=U.dtype, device=sim_params.device)

        max_lam = sim_params.lams[sim_params.weights.argmax()]
        
        for i, lam in enumerate(sim_params.lams):
            t = self.thickness
            n_slices = int(t // slice_thickness)
            U_i_slice = U[i, :, :].unsqueeze(0)
            for j in range(n_slices):
                t_slice = slice_thickness
                if j == n_slices - 1:
                    t_slice = t - j * slice_thickness
                slice_element = self.__copy__()
                slice_element.thickness = t_slice

                transmission = slice_element.transmission(lam, max_lam, sim_params)
                U_lam = U_i_slice * transmission
                U_lamz = angular_spectrum_propagation(U_lam, lam, t_slice, sim_params.dx, sim_params.device)
                U_i_slice = U_lamz
            U_f[i, :, :] = U_i_slice.squeeze(0)
        return U_f
