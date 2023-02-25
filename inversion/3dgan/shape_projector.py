# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from camera_utils import GaussianCameraPoseSampler
from torch_utils import misc
from training.triplane import SRPosedGenerator

from tqdm import tqdm
import mrcfile
import pickle

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target_noise', 'target_noise_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=0, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    target_noise_fname: str,
    noise_mode: str,
    outdir: str,
    seed: int
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    #with dnnlib.util.open_url(network_pkl) as fp:
    #    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    G = torch.load(network_pkl)
    G = G.requires_grad_(False).to(device)


    # Load optimized noise.
    f = open(target_noise_fname,'rb')
    noise_dict = pickle.load(f)
    projected_w = torch.as_tensor(noise_dict['projected_w'].astype(np.float32), device=device)
    noise_bufs = noise_dict['noise_bufs']

    G.backbone.synthesis.b4.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b4.conv1.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b8.conv0.register_buffer('noise_const', torch.as_tensor(noise_bufs["b8.conv0.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b8.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b8.conv1.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b16.conv0.register_buffer('noise_const', torch.as_tensor(noise_bufs["b16.conv0.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b16.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b16.conv1.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b32.conv0.register_buffer('noise_const', torch.as_tensor(noise_bufs["b32.conv0.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b32.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b32.conv1.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b64.conv0.register_buffer('noise_const', torch.as_tensor(noise_bufs["b64.conv0.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b64.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b64.conv1.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b128.conv0.register_buffer('noise_const', torch.as_tensor(noise_bufs["b128.conv0.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b128.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b128.conv1.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b256.conv0.register_buffer('noise_const', torch.as_tensor(noise_bufs["b256.conv0.noise_const"].astype(np.float32), device=device))
    G.backbone.synthesis.b256.conv1.register_buffer('noise_const', torch.as_tensor(noise_bufs["b256.conv1.noise_const"].astype(np.float32), device=device))

    # Sample generator sigma values.
    os.makedirs(outdir, exist_ok=True)
    voxel_resolution = 512
    max_batch=1000000

    samples, _, _ = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=0.7)
    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                planes = G.backbone.synthesis(projected_w, noise_mode=noise_mode)
                planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

                G.rendering_kwargs['depth_resolution'] = 256
                G.rendering_kwargs['depth_resolution_importance'] = 256
                sigma = G.renderer.run_model(planes, G.decoder, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], G.rendering_kwargs)['sigma']
                sigmas[:, head:head+max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    with mrcfile.new_mmap(f'{outdir}/seed{seed:04d}.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        mrc.data[:] = sigmas

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
