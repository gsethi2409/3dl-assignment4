import numpy as np
import torch
from ray_utils import RayBundle

def phong(
    normals,
    view_dirs, 
    light_dir,
    params,
    colors
):
    # TODO: Implement a simplified version Phong shading
    # Inputs:
    #   normals: (N x d, 3) tensor of surface normals
    #   view_dirs: (N x d, 3) tensor of view directions
    #   light_dir: (3,) tensor of light direction
    #   params: dict of Phong parameters
    #   colors: (N x d, 3) tensor of colors
    # Outputs:
    #   illumination: (N, 3) tensor of shaded colors
    #
    # Note: You can use torch.clamp to clamp the dot products to [0, 1]
    # Assume the ambient light (i_a) is of unit intensity 
    # While the general Phong model allows rerendering with multiple lights, 
    # here we only implement a single directional light source of unit intensity
    # pass

    light_dir = light_dir.repeat(normals.shape[0], 1)
    light_dir = torch.nn.functional.normalize(light_dir, dim=-1).view(-1, 3)
    
    L_dot_N = torch.clamp(torch.sum((normals * light_dir), axis=1).reshape(-1,1), 0, 1)
    
    Rm = (2 * L_dot_N * normals) - light_dir
    
    Rm_dot_V = torch.sum((Rm * view_dirs), axis=1).reshape(-1,1)
    Ip = (params['ka'] * colors) + \
            params['kd'] * torch.clamp(((L_dot_N) * colors), 0, 2) + \
                params['ks'] * torch.clamp((torch.pow(Rm_dot_V, params['n']) * colors),0,1)
    return Ip

relighting_dict = {
    'phong': phong
}