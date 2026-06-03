import torch # type: ignore
import numpy as np # type: ignore
from typing import Callable
from .simparams import SimParams
from .elements import ZonePlate
import copy

def create_tracking_objective_function(
    beta: float, 
    forward_model: Callable, 
    sim_params: SimParams, 
    opt_params: dict, 
    forward_model_args: tuple,
    obj_values: list,
    x_values: list,
    intermediate_tensors: list,
    Nthreshold: int = None
    ) -> Callable:
    """
    Create a tracking objective function that logs objective values, parameter vectors, 
    and intermediate tensors during optimization.
    
    Args:
        beta: Beta value for heaviside projection
        forward_model: Forward model function
        sim_params: Simulation parameters
        opt_params: Optimization parameters
        forward_model_args: Additional arguments for forward model
        obj_values: List to track objective values
        x_values: List to track parameter vectors
        intermediate_tensors: List to track intermediate tensors
        Nthreshold: Number of parameters to apply thresholding to (if None, applies to all)
    """
    def tracking_objective_function(x, grad):
        # Convert to PyTorch tensor to call forward model directly
        zero = torch.zeros(0, dtype=sim_params.dtype, device=sim_params.device)
        g = torch.tensor(x, dtype=zero.real.dtype, requires_grad=True, device=sim_params.device)
        
        # Apply same preprocessing as in create_objective_function
        if Nthreshold is not None:
            # only apply preprocessing to parameters with indices below Nthreshold
            g_filtered = g[:Nthreshold]
            g_filtered = density_filtering(g_filtered, opt_params["filter_radius"], sim_params)
            g_thresholded = heaviside_projection(g_filtered, beta=beta)
            g_physical = g_thresholded.view(-1)
            g_physical = torch.cat((g_physical, g[Nthreshold:]), dim=0)
        else:
            g_filtered = density_filtering(g, opt_params["filter_radius"], sim_params)
            g_thresholded = heaviside_projection(g_filtered, beta=beta)
            g_physical = g_thresholded.view(-1)
        
        g_physical.retain_grad()
        
        # Call forward model directly to get tuple result
        forward_result = forward_model(g_physical, sim_params, opt_params, *forward_model_args)
        
        if isinstance(forward_result, tuple):
            obj_val = forward_result[0]
            intermediate_tensor = forward_result[1]
        else:
            obj_val = forward_result
            intermediate_tensor = None
        
        # Handle gradients if needed
        if grad.size > 0:
            obj_val.backward()
            grad[:] = g_physical.grad.detach().cpu().numpy()
        
        # Track values
        obj_values.append(obj_val.item())
        x_values.append(x.copy())
        if intermediate_tensor is not None:
            intermediate_tensors.append(intermediate_tensor.detach().cpu().numpy())
        
        return obj_val.item()
    
    return tracking_objective_function


def zp_init(
        lam: float, 
        f: float, 
        min_feature_size: float, 
        n: int,
        sim_params: SimParams
) -> np.ndarray:
    zone_plate = ZonePlate(
        name = "zp_init", 
        thickness = 1, 
        f = f,
        min_feature_size = min_feature_size, 
        elem_map = [np.array([0, np.inf]), np.array([1., 1.])], 
        gap_map = [np.array([0, np.inf]), np.array([1 + 1j*np.inf, 1 + 1j*np.inf])]
    )

    zp_trans = zone_plate.transmission(lam, lam, sim_params).abs()
    zp_init = torch.where(zp_trans > 0.5, 1.0, 0.0).cpu().reshape(sim_params.Nx)[::n]
    zp_init = zp_init[:zp_init.shape[0]//2].numpy()
    return zp_init
