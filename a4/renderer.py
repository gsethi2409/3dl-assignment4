import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase
import pdb

from a4.lighting_functions import relighting_dict

# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q1): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        # pass

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        points = origins.clone()
        eps_threshold = 1e-4
        mask = torch.zeros((origins.shape[0],1), dtype=torch.bool)
        
        for i in range(self.max_iters):
            distance = implicit_fn.get_distance(torch.Tensor(points).to(device))
            points += (distance * directions) 
            mask[distance < eps_threshold] = 1
        
        return points, mask

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir = None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q3): Convert signed distance to density with alpha, beta parameters
    # pass

    # psi_beta = torch.zeros_like(signed_distance)
    # inds1 = torch.where(-signed_distance <= 0)
    # psi_beta[inds1] = 0.5 * torch.exp(-signed_distance[inds1] / beta)
    # inds2 = torch.where(-signed_distance > 0)
    # psi_beta[inds2] = 1 - 0.5 * torch.exp(signed_distance[inds2] / beta)
    # sigma = alpha * psi_beta

    # q5.3
    e = -signed_distance * 50
    sigma = 50 * torch.exp(e) / (1 + torch.exp(e))**2

    return sigma

class VolumeSDFRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (Q3): Copy code from VolumeRenderer._compute_weights
        # pass

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        emission = 1 - torch.exp(-rays_density * deltas)
        num_dirs = rays_density.shape[0]
        num_pts = rays_density.shape[1]
        
        T_initial = [torch.ones((num_dirs,1)).to(device)]
        for ray_point in range(num_pts-1):
            T_initial.append(T_initial[-1]* torch.exp(-rays_density[:,ray_point]* deltas[:,ray_point]))

        T_chain = torch.stack(T_initial,dim=1)
        weights = T_chain * emission
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_color: torch.Tensor
    ):
        # TODO (Q3): Copy code from VolumeRenderer._aggregate
        # pass
        rays_features = (weights * rays_color)
        feature = torch.sum(rays_features,axis=1)
        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir = None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            # density = None # TODO (Q3): convert SDF to density
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)
            if light_dir is not None:
                normals = implicit_fn.get_surface_normal(cur_ray_bundle.sample_points)
                view_dirs = -cur_ray_bundle.directions.repeat(n_pts, 1)
                geometry_color[color.sum(dim=1) > 1e-3] = torch.tensor([0.7, 0.7, 1.0]).to(color.device)
                params = {"ka": self.cfg.relighting_function.ka, 
                        "kd": self.cfg.relighting_function.kd, 
                        "ks": self.cfg.relighting_function.ks,  
                        "n": self.cfg.relighting_function.n # This is analogous to alpha in the Phong model
                }
                color = relighting_dict[self.cfg.relighting_function.type](normals, view_dirs, light_dir, params, color)
                geometry_color = relighting_dict[self.cfg.relighting_function.type](normals, view_dirs, light_dir, params, geometry_color) 
                geometry_color = self._aggregate(
                    weights,
                    geometry_color.view(-1, n_pts, geometry_color.shape[-1])
                )

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}

