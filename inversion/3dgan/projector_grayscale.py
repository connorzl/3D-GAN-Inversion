# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn
import torch.nn.functional as F
import click

import dnnlib
import legacy
import pickle

def project(
    G,
    target_img: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_pose: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
):
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    masked_indices = np.where(target_mask == 0)
    mask_x = masked_indices[0]
    mask_y = masked_indices[1]

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), target_pose.repeat(w_avg_samples, 1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target_img.unsqueeze(0).to(device).to(torch.float32)
    min_val = torch.min(target_images)
    max_val = torch.max(target_images)
    target_images = target_images.repeat(1, 3, 1, 1)
    target_images = (target_images - min_val) / (max_val - min_val)
    target_images = target_images * 255
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')

    # Mask target
    target_images[:, :, mask_x, mask_y] = 0
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt, target_pose] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, target_pose, noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        synth_images = 0.3 * synth_images[:, 0, :, :] + 0.59 * synth_images[:, 1, :, :] + 0.11 * synth_images[:, 2, :, :]
        min_val = torch.min(synth_images)
        max_val = torch.max(synth_images)
        synth_images = synth_images.repeat(1, 3, 1, 1)
        synth_images = (synth_images - min_val) / (max_val - min_val)
        synth_images = synth_images * 255

        # Mask generated.
        synth_images[:, :, mask_x, mask_y] = 0

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.backbone.mapping.num_ws, 1]), noise_bufs

#----------------------------------------------------------------------------

def normalize_rgb(img):
    synth_image = (img + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return synth_image

def normalize_depth(img):
    depth_image = -img.permute(0, 2, 3, 1)[0]
    depth_image = depth_image.cpu().numpy().astype(np.float32)
    lo, hi = depth_image.min(), depth_image.max()
    depth_image = (depth_image - lo) * (255 / (hi - lo))
    depth_image = np.rint(depth_image).clip(0, 255).astype(np.uint8)
    return depth_image

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target_image', 'target_image_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--target_pose', 'target_pose_fname', help='Target pose file to project to', required=True, metavar='FILE')
@click.option('--target_mask', 'target_mask_fname', help='Target mask file to project to', default="", show_default=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=0, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--steps-per-frame',        help='Optimization steps per video frame', type=int, default=5, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_image_fname: str,
    target_pose_fname: str,
    target_mask_fname: str,
    outdir: str,
    save_video: bool,
    steps_per_frame: int,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.
    Examples:
    \b
    python projector.py --outdir=out --target_image=~/mytargetimg.png --target_pose=~/mytargetpose.json \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_image_fname)
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_uint8 = np.expand_dims(target_uint8, axis=2)


    # Load target pose.
    target_pose = np.loadtxt(target_pose_fname, delimiter=',').astype(np.float32)
    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    target_pose = torch.tensor(target_pose, device=device).unsqueeze(0)

    # Load target mask.
    target_mask = np.array(PIL.Image.open(target_mask_fname), dtype=np.uint8) / 255

    # Optimize projection.
    start_time = perf_counter()
    # num_steps x num_ws x z_dim
    projected_w_steps, noise_bufs = project(
        G,
        target_img=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        target_pose=target_pose, 
        target_mask=target_mask,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    target_rgb_pil = PIL.Image.open(target_image_fname).convert('RGB')
    w, h = target_rgb_pil.size
    s = min(w, h)
    target_rgb_pil = target_rgb_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_rgb_pil = target_rgb_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_rgb = np.array(target_rgb_pil, dtype=np.uint8)

    os.makedirs(outdir, exist_ok=True)
    if save_video:
        rgb_video = imageio.get_writer(f'{outdir}/rgb_proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        depth_video = imageio.get_writer(f'{outdir}/depth_proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for i, projected_w in enumerate(projected_w_steps):
            if i % steps_per_frame == 0:
                synth_result = G.synthesis(projected_w.unsqueeze(0), target_pose, noise_mode='const')
                rgb_video.append_data(np.concatenate([target_rgb, normalize_rgb(synth_result['image'])], axis=1))
                depth_video.append_data(normalize_depth(synth_result['image_depth']))
        rgb_video.close()
        depth_video.close()

    # Save final projected frame.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), target_pose, noise_mode='const')['image']
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')

    # Save optimized noise.
    for noise_buf in noise_bufs:
        noise_bufs[noise_buf] = noise_bufs[noise_buf].detach().cpu().numpy()
    optimized_dict = {
        'projected_w': projected_w.unsqueeze(0).cpu().numpy(),
        'noise_bufs': noise_bufs
    }
    with open(f'{outdir}/optimized_noise_dict.pickle', 'wb') as handle:
        pickle.dump(optimized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
    

