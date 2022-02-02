import torch.nn as nn
import torch.nn.functional as F
import torch
from im2scene.common import (
    arange_pixels, image_points_to_world, origin_to_world
)
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from im2scene.camera import get_camera_mat, get_random_pose, get_camera_pose
import torch, gc

class Generator(nn.Module):
    ''' GIRAFFE Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        decoder (nn.Module): decoder network
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        bounding_box_generaor (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
         (0 - 1)
        sample_object-existance (bool): whether to sample the existance
            of objects; only used for clevr2345
        use_max_composition (bool): whether to use the max
            composition operator instead
    '''

    def __init__(self, device, batch_size=None, z_dim=256, decoder=None,
                 range_u=None, range_v=None, n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 use_max_composition=False, **kwargs):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.fov = fov
        self.z_dim = z_dim
        self.use_max_composition = use_max_composition
        self.batch_size = batch_size
        self.camera_matrix = get_camera_mat(fov=fov).to(device)

        if decoder is not None:
            self.decoder = decoder.to(device)
        else:
            self.decoder = None
        if neural_renderer is not None:
            self.neural_renderer = neural_renderer.to(device)
        else:
            self.neural_renderer = None

    def forward(self, latent_codes=None, camera_matrices=None,
                mode="training", it=0,
                return_alpha_map=False):

        batch_size = self.batch_size
        if latent_codes is None:
            latent_codes = self.get_latent_codes(batch_size)

        if camera_matrices is None:
            camera_matrices = self.get_random_camera(batch_size)

        latent_codes = latent_codes[0].squeeze(), latent_codes[1].squeeze()


        rgb_v = self.volume_render_image(
            latent_codes, camera_matrices, 
            mode=mode, it=it)
        rgb = self.neural_renderer(rgb_v)
        return rgb


    def get_latent_codes(self, batch_size=32, tmp=1.):
        z_dim = self.z_dim

        def sample_z(x): return self.sample_z(x, tmp=tmp)
        z_shape_obj = sample_z((batch_size, 1, z_dim))
        z_app_obj = sample_z((batch_size, 1, z_dim))

        return z_shape_obj, z_app_obj

    def sample_z(self, size, to_device=True, tmp=1.):
        z = torch.randn(*size) * tmp
        if to_device:
            z = z.to(self.device)
        return z

    def get_vis_dict(self):
        batch_size = self.batch_size
        vis_dict = {
            'latent_codes': self.get_latent_codes(batch_size),
            'camera_matrices': self.get_random_camera(batch_size),
        }
        return vis_dict

    def get_random_camera(self, batch_size=32, to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_random_pose(
            self.range_u, self.range_v, self.range_radius, batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def get_camera(self, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32,
                   to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_camera_pose(
            self.range_u, self.range_v, self.range_radius, val_u, val_v,
            val_r, batch_size=batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat



    def get_evaluation_points(self, pixels_world, camera_world, di):
                              
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        ray_i = pixels_world - camera_world

        p_i = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)

        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(
                                         1, -1, 1), torch.arange(ns).reshape(
                                             1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)
        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights


    def volume_render_image(self, latent_codes, camera_matrices,
                            mode='training',
                            it=0, return_alpha_map=False):
        res = self.resolution_vol
        device = self.device
        n_steps = self.n_ray_samples
        n_points = res * res
        depth_range = self.depth_range
        batch_size = latent_codes[0].shape[0]
        z_shape_obj, z_app_obj = latent_codes

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size,
                               invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.
        # Project to 3D world
        pixels_world = image_points_to_world(
            pixels, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        camera_world = origin_to_world(
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
            torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        if mode == 'training':
            di = self.add_noise_to_interval(di)

        n_boxes = latent_codes[0].shape[1]
        feat, sigma = [], []

        p_i, r_i = self.get_evaluation_points(
            pixels_world, camera_world, di)
        z_shape_i, z_app_i = z_shape_obj, z_app_obj

        feat_i, sigma_i = self.decoder(p_i, r_i, z_shape_i, z_app_i)

        if mode == 'training':
            # As done in NeRF, add noise during training
            sigma_i += torch.randn_like(sigma_i)

        # Mask out values outside
        padd = 0.1
        mask_box = torch.all(
            p_i <= 1. + padd, dim=-1) & torch.all(
                p_i >= -1. - padd, dim=-1)
        sigma_i[mask_box == 0] = 0.

        # Reshape
        sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
        feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)
        sigma = F.relu(sigma_i)
        feat = feat_i

        # Composite
        sigma_sum, feat_weighted = self.composite_function(sigma, feat)

        # Get Volume Weights
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)

        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, res, res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        return feat_map
