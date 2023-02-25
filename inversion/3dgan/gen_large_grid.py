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
from PIL import Image, ImageDraw, ImageFont
import torch

import legacy
from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import SRPosedGenerator

#----------------------------------------------------------------------------

def make_image_grid(img, fname, drange, grid_size):
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
        return Image.fromarray(img[:, :, 0], 'L')
    if C == 3:
        return Image.fromarray(img, 'RGB')

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int],
    reload_modules: bool,
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

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if reload_modules:
        print("Reloading Modules!")
        G_new = SRPosedGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)

    # Generate images.
    all_imgs = []
    if len(seeds) == 1000:
        gw = 40 
        gh = 25
    elif len(seeds) == 100:
        gw = 10
        gh = 10
    elif len(seeds) == 500:
        gw = 25
        gh = 20
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        label = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = G.mapping(z, label, truncation_psi=truncation_psi)
        img = G.synthesis(ws, label, noise_mode=noise_mode)['image']

        all_imgs.append(img.squeeze(0).cpu().numpy())

    all_imgs = np.stack(all_imgs)
    W, H = all_imgs.shape[-2], all_imgs.shape[-1]
    base_grid = make_image_grid(all_imgs, os.path.join(outdir, f'seed%d-%d.png' % (seeds[0], seeds[-1])), drange=[-1, 1], grid_size=(gw, gh))

    txt = Image.new("RGBA", base_grid.size, (0,0,0,0))
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    d = ImageDraw.Draw(txt)
   
    for row_idx in range(gh):
        for col_idx in range(gw):
            seed_idx = row_idx * gw + col_idx
            print('Write text for seed %d (%d/%d) ...' % (seeds[seed_idx], seed_idx, len(seeds)))
            d.text((col_idx * W, row_idx * H), f"%04d" % seeds[seed_idx], font=fnt, fill=(0,0,0,255))

    out = Image.alpha_composite(base_grid.convert('RGBA'), txt)
    out.save(os.path.join(outdir, f'seed%d-%d.png' % (seeds[0], seeds[-1])))

#         labels = [
# [0.9849246144294739, -0.032208047807216644, -0.1699592024087906, 0.43596652608351827, -0.007078535854816437, -0.9891948103904724, 0.14643636345863342, -0.36647229261129954, -0.1728391796350479, -0.1430257111787796, -0.9745102524757385, 2.639248235176617, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0],
#             [0.9995502233505249, 0.012934454716742039, -0.027058573439717293, 0.06911902701402448, 0.007681829854846001, -0.9825384020805359, -0.18590129911899567, 0.4644204774393026, -0.028990620747208595, 0.185609832406044, -0.9821957945823669, 2.6588599399441266, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0],
# [0.940233051776886, -0.027783220633864403, 0.3393963575363159, -0.8656389060258219, 0.0368637889623642, -0.9825047254562378, -0.18255263566970825, 0.4822158896840353, 0.33853039145469666, 0.1841534674167633, -0.9227592945098877, 2.5116005096572693, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0],
# [0.9399276971817017, -0.07601111382246017, 0.33280372619628906, -0.8485870508177907, 0.01686999201774597, -0.9633619785308838, -0.26767364144325256, 0.6992135693711592, 0.34095659852027893, 0.2572082579135895, -0.904208242893219, 2.4659684510535835, 0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0],

#         ]
#         imgs = []
#         for label in labels:
#             forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
#             label = torch.tensor(label, device=device).unsqueeze(0)
#             forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

#             ws = G.mapping(z, forward_label, truncation_psi=truncation_psi)
#             img = G.synthesis(ws, label, noise_mode=noise_mode)['image']

#             img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#             imgs.append(img)

        #img = torch.cat(imgs, dim=2)
        #PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
