import torch
import numpy as np


def heat_residual(T, S, coords, t, mat_props):
    """
    Calculate the residual of the heat equation with moving laser source
    
    Heat equation: ρc_p∂T/∂t = ∇·(k∇T) + 2ηP/(πr_0²)exp(-2r²/r_0²) - H_m∂f_s/∂t
    
    Args:
        T (torch.Tensor): Temperature field
        S (torch.Tensor): Process parameters
        coords (torch.Tensor): Spatial coordinates (x, y, z)
        t (torch.Tensor): Time coordinates
        mat_props (dict): Material properties
            - rho: Density
            - cp: Specific heat capacity
            - k: Thermal conductivity
            - eta: Absorption coefficient
            - P: Laser power
            - r0: Laser beam radius
            - Hm: Latent heat of melting
            - Ts: Solidus temperature
            - Tl: Liquidus temperature
    
    Returns:
        torch.Tensor: Residual of heat equation
    """
    # Extract material properties
    rho = mat_props['rho']
    cp = mat_props['cp']
    k = mat_props['k']
    eta = mat_props['eta']
    P = S[:, 0]  # Laser power from scan vector
    r0 = mat_props['r0']
    Hm = mat_props['Hm']
    Ts = mat_props['Ts']
    Tl = mat_props['Tl']
    
    # Create computational graph for automatic differentiation
    coords.requires_grad_(True)
    t.requires_grad_(True)
    
    # Compute gradients for temperature
    T_pred = T(torch.cat([coords, t], dim=1))
    
    # Time derivative: ∂T/∂t
    dTdt = torch.autograd.grad(
        T_pred.sum(), t, create_graph=True
    )[0]
    
    # Spatial derivatives: ∇T and ∇²T (Laplacian)
    grad_T = torch.autograd.grad(
        T_pred.sum(), coords, create_graph=True
    )[0]
    
    # For Laplacian, we need to compute the divergence of the gradient
    laplacian_T = 0
    for i in range(3):  # For each dimension (x, y, z)
        second_deriv = torch.autograd.grad(
            grad_T[:, i].sum(), coords, create_graph=True
        )[0][:, i]
        laplacian_T += second_deriv
    
    # Calculate laser heat source term
    # Laser position depends on time and scan speed (v)
    v = S[:, 1]  # Scan speed from scan vector
    
    # Laser position at time t assuming it moves along x-axis
    x_laser = v * t.squeeze()
    y_laser = torch.zeros_like(x_laser)
    z_laser = torch.zeros_like(x_laser)
    
    # Calculate distance to laser center
    r_squared = (coords[:, 0] - x_laser)**2 + \
               (coords[:, 1] - y_laser)**2 + \
               (coords[:, 2] - z_laser)**2
    
    # Gaussian laser heat source term
    q_laser = 2*eta*P/(np.pi*r0**2) * torch.exp(-2*r_squared/r0**2)
    
    # Phase change term (latent heat)
    # Approximate the solid fraction with a smooth function
    fs = 0.5 * (1 + torch.tanh((Tl - T_pred) / (Tl - Ts) * 10))
    dfsdT = torch.autograd.grad(
        fs.sum(), T_pred, create_graph=True
    )[0]
    dfsdt = dfsdT * dTdt
    
    # Assemble the heat equation
    residual = rho*cp*dTdt - k*laplacian_T - q_laser + Hm*dfsdt
    
    return residual


def stress_residual(sigma, coords, mat_props):
    """
    Calculate the residual of the static equilibrium equation for stress
    
    Static equilibrium: ∇·σ = 0
    
    Args:
        sigma (torch.Tensor): Stress tensor field
        coords (torch.Tensor): Spatial coordinates (x, y, z)
        mat_props (dict): Material properties
    
    Returns:
        torch.Tensor: Residual of static equilibrium equation
    """
    # Create computational graph for automatic differentiation
    coords.requires_grad_(True)
    
    # Predict the full stress tensor (symmetric 3x3 tensor)
    stress_tensor = sigma(coords)
    
    # Reshape the output to get the 6 independent components of the stress tensor
    # [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]
    stress = stress_tensor.view(-1, 6)
    
    # Compute the divergence of the stress tensor (∇·σ)
    div_sigma = torch.zeros(coords.shape[0], 3, device=coords.device)
    
    # Loop over each dimension
    for i in range(3):
        # Extract stress components in each direction
        if i == 0:  # x-direction
            s_i = torch.cat([
                stress[:, 0].unsqueeze(1),  # σ_xx
                stress[:, 3].unsqueeze(1),  # σ_xy
                stress[:, 5].unsqueeze(1)   # σ_xz
            ], dim=1)
        elif i == 1:  # y-direction
            s_i = torch.cat([
                stress[:, 3].unsqueeze(1),  # σ_xy
                stress[:, 1].unsqueeze(1),  # σ_yy
                stress[:, 4].unsqueeze(1)   # σ_yz
            ], dim=1)
        else:  # z-direction
            s_i = torch.cat([
                stress[:, 5].unsqueeze(1),  # σ_xz
                stress[:, 4].unsqueeze(1),  # σ_yz
                stress[:, 2].unsqueeze(1)   # σ_zz
            ], dim=1)
        
        # Compute derivative of the stress components
        for j in range(3):
            grad_s = torch.autograd.grad(
                s_i[:, j].sum(), coords, create_graph=True
            )[0][:, j]
            div_sigma[:, i] += grad_s
    
    # The residual is the divergence which should be zero (∇·σ = 0)
    return div_sigma


def compute_physics_loss(model, S_batch, coords_batch, t_batch, mat_props, lambda_heat=1.0, lambda_stress=1.0):
    """
    Compute the combined physics loss for the PINN
    
    Args:
        model: The PINN model
        S_batch: Process parameters batch
        coords_batch: Spatial coordinates batch
        t_batch: Time coordinates batch
        mat_props: Material properties
        lambda_heat: Weight for heat equation residual
        lambda_stress: Weight for stress equation residual
    
    Returns:
        torch.Tensor: Combined physics loss
    """
    # Define a temperature model as a function of coordinates and time
    def temperature_model(x):
        inp = torch.cat([S_batch, x], dim=1)
        out = model(inp)
        T = out[:, 0]  # Assuming first output is temperature
        return T.unsqueeze(1)
    
    # Define a stress model as a function of coordinates
    def stress_model(x):
        inp = torch.cat([S_batch, x], dim=1)
        out = model(inp)
        stress = out[:, 1:]  # Assuming remaining outputs are stress components
        return stress
    
    # Compute residuals
    heat_res = heat_residual(temperature_model, S_batch, coords_batch, t_batch, mat_props)
    stress_res = stress_residual(stress_model, coords_batch, mat_props)
    
    # Compute losses
    heat_loss = torch.mean(heat_res**2)
    stress_loss = torch.mean(torch.sum(stress_res**2, dim=1))
    
    # Combine losses
    physics_loss = lambda_heat * heat_loss + lambda_stress * stress_loss
    
    return physics_loss