import os
import re
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import scipy.interpolate
import PIL
from PIL import Image
import click
from tqdm import tqdm

import dnnlib
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import SRPosedGenerator

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

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def parse_range(s: Union[str, List[int]]) -> List[int]:
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

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        Image.fromarray(img, 'RGB').save(fname)

@click.command()
@click.option('--network', required=True)
@click.option('--seeds', type=parse_range, required=True)
@click.option('--grid', type=parse_tuple, default=(1, 5))
@click.option('--truncation', type=float, default=1)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--reload_modules', type=bool, default=True)
@click.option('--outdir', type=str, required=True)
def generate_interpolate_image(
    network: str,
    seeds: List[int],
    grid: Tuple[int, int],
    truncation: float,
    noise_mode: str,
    reload_modules: bool,
    outdir: str,
):
    print('Loading network from %s' % network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if reload_modules:
        print('Reloading module')
        G_new = SRPosedGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    # set neural rendering resolution for new model
    G.neural_rendering_resolution = 128
    
    grid_h, grid_w = grid[0], grid[1]
    assert grid_h == 1, "only supports single row grid"

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    ws = G.mapping(z=zs, c=c, truncation_psi=truncation)
  
    xp = [0, grid_w - 1]
    fp = ws
    imgs = []
    interp = scipy.interpolate.interp1d(xp, fp.cpu().numpy(), axis=0)

    max_batch = 1000000
    voxel_resolution = 512
    for col_idx in range(grid_w):
        interp_w = torch.from_numpy(interp(col_idx)).to(device)
        img = G.synthesis(ws=interp_w.unsqueeze(0), c=c[0].unsqueeze(0), noise_mode='const')['image'][0]
        imgs.append(img.cpu().numpy())

        # generate shapes
        print('Generating shape for index %d / %d ...' % (col_idx, grid_w))
        
        samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'])
        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], interp_w.unsqueeze(0), truncation_psi=truncation, noise_mode=noise_mode)['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)
        
        pad = int(30 * voxel_resolution / 256)
        pad_top = int(35 * voxel_resolution / 256)
        sigmas[:pad] = 0
        sigmas[-pad:] = 0
        sigmas[:, :pad_top] = 0
        sigmas[:, -pad_top:] = 0
        sigmas[:, :, :pad] = 0
        sigmas[:, :, -pad:] = 0

        with mrcfile.new_mmap(f'{outdir}/{col_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas

    os.makedirs(outdir, exist_ok=True)
    save_image_grid(np.stack(imgs), os.path.join(outdir, f'grid_%d_%d.png' % (seeds[0], seeds[1])), drange=[-1, 1], grid_size=(grid_w, grid_h))

    for idx in range(len(imgs)):
        img = (imgs[idx].transpose(1, 2, 0) * 127.5 + 128).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{idx:04d}_image.png')

if __name__ == '__main__':
    generate_interpolate_image()
