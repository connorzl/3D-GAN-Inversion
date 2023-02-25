# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import pickle
import json
from torchvision.utils import save_image

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import SRPosedGenerator
#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

#----------------------------------------------------------------------------

def gen_interp_video(G, ws, pose, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0], device=device)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    #ws = G.mapping(z=zs, c=c, truncation_psi=psi)

    _ = G.synthesis(ws[:1], c[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    G.rendering_kwargs['depth_resolution'] *= 4
    G.rendering_kwargs['depth_resolution_importance'] *= 4
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    #all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                #cam2world_pose = LookAtPoseSampler.sample(3.14/2 + 0.35 * np.sin(2 * 3.14 * frame_idx / w_frames),
                #                                          3.14/2 - 0.2 + 0.25 * np.cos(2 * 3.14 * frame_idx / w_frames),
                #                                        torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
                #all_poses.append(cam2world_pose.detach().cpu().numpy())
                #intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                #c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                c = pose

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                
                entangle = 'camera'
                if entangle == 'conditioning':
                    c_forward = torch.cat([LookAtPoseSampler.sample(3.14/2,
                                                                    3.14/2,
                                                                    camera_lookat_point,
                                                                    radius=2.7, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    img = G.synthesis(ws=w_c, c=c_forward, noise_mode='const')[image_mode][0]
                elif entangle == 'camera':
                    img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')['image'][0]
                    #img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(512,512), mode='nearest').squeeze(0)
                elif entangle == 'both':
                    w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    img = G.synthesis(ws=w_c, c=c[0:1], noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)
    print(all_poses.shape)
    with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
        np.save(f, all_poses)

    #all_poses = np.stack(all_poses, axis=0)
    #np.save("/media/data6/connorzl/pigan/S-GAN/stylegan3/pti_inversion/embeddings/poses.npy", all_poses)

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target_noise', 'target_noise_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=360)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc_cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)


def generate_images(
    network_pkl: str,
    target_noise_fname: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    #with dnnlib.util.open_url(network_pkl) as f:
    #    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G = torch.load(network_pkl)
    G = G.requires_grad_(False).to(device)

    if reload_modules:
        print("Reloading Modules!")
        G_new = SRPosedGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    
    f = open(target_noise_fname,'rb')
    noise_dict = pickle.load(f)
    ws = torch.as_tensor(noise_dict['projected_w'].astype(np.float32), device=device)
    #noise_bufs = noise_dict['noise_bufs']
    
    #ws = np.load(target_noise_fname)
    #ws = torch.from_numpy(ws).to(device)

    f = open("pretrained_models/cameras.json")
    all_poses = json.load(f)
    f.close()

    G.rendering_kwargs['depth_resolution'] *= 4
    G.rendering_kwargs['depth_resolution_importance'] *= 4

    w = ws[0]
    target_pose = np.asarray(all_poses["taylor2.jpg"]['pose']).astype(np.float32)
    o = target_pose[0:3, 3]
    o = 2.7 * o / np.linalg.norm(o)
    target_pose[0:3, 3] = o
    target_pose = np.reshape(target_pose, -1)    
    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    c = torch.tensor(target_pose).unsqueeze(0).to(device)
    
    img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')['image'][0]
    img = (img * 0.5 + 0.5).clamp(0, 1)
    save_image(img, "pretrained_models/taylor2_inversion.png")
    
    """
    for i in range(ws.shape[0]):
        print("wei:", i)
        w = ws[i]

        file_id = "obama" + str(i+1).zfill(3) + ".jpg"
        target_pose = np.asarray(all_poses[file_id]['pose']).astype(np.float32)
        o = target_pose[0:3, 3]
        o = 2.7 * o / np.linalg.norm(o)
        target_pose[0:3, 3] = o
        target_pose = np.reshape(target_pose, -1)    
        intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
        target_pose = np.concatenate([target_pose, intrinsics])
        c = torch.tensor(target_pose).unsqueeze(0).to(device)
        
        img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')['image'][0]
        img = (img * 0.5 + 0.5).clamp(0, 1)
        save_image(img, "pti_inversion/logs/puppet_aged/output_1000/" + str(i).zfill(4) + "_posed.png")
    """ 
    """
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
    """

    #gen_interp_video(G=G, ws=projected_w, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi)
    #gen_interp_video(G=G, ws=None, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
