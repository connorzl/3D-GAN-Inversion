import sys
import os
sys.path.append("../3dgan")

import numpy as np
import torch
import json
from configs import paths_config, global_config
from utils.models_utils import toogle_grad, load_3dgan, load_tuned_G
import torch.nn.functional as F
from camera_utils import LookAtPoseSampler
import skimage.io
import skimage.transform
from kornia import morphology as morph
from kornia.filters import gaussian_blur2d
from mpl_toolkits import mplot3d
from tqdm import tqdm
from skimage.morphology.convex_hull import convex_hull_image
from glob import glob
from training import triplane

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_logs_path", default="")
parser.add_argument("--input_data_path", default="")
parser.add_argument("--input_pose_path", default="")
parser.add_argument("--target_pose_path", default="")

np.random.seed(1989)
torch.manual_seed(1989)

def get_json_pose(json_file, key):
    f = open(json_file)
    target_pose = np.asarray(json.load(f)[key]['pose']).astype(np.float32)
    f.close()
    o = target_pose[0:3, 3]
    o = 2.7 * o / np.linalg.norm(o)
    target_pose[0:3, 3] = o
    target_pose = np.reshape(target_pose, -1)

    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    target_pose = torch.tensor(target_pose, device=global_config.device).unsqueeze(0)
    return target_pose

def get_numpy_pose(numpy_file):
    target_pose = np.load(numpy_file).astype(np.float32)
    target_pose = np.reshape(target_pose, -1)

    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    target_pose = torch.tensor(target_pose, device=global_config.device).unsqueeze(0)
    return target_pose


def sample_w(G, pose):
    # sample an image
    z_sample = np.random.RandomState(123).randn(1, G.z_dim)
    z_sample = torch.from_numpy(z_sample).to(global_config.device)
    w = G.mapping(z_sample, pose, truncation_psi=0.7)  # [N, L, C]
    w = torch.from_numpy(np.load('seed0000.npy')).to(global_config.device)
    return w


def get_init_mask_coords(mask=None):
    basename = os.path.basename(args.input_logs_path).split('_')[0]
    default_pose = get_json_pose(args.input_pose_path, f'{basename}.jpg')

    with torch.no_grad():
        out = G.synthesis(w_full[[0]], default_pose, noise_mode='const', force_fp32=True)
        depth = F.interpolate(out['image_depth'], (512, 512))
        depth = depth.reshape(-1)[None, ..., None]

        # use default face mask
        if mask is None:
            mask = sorted(glob(os.path.join(args.input_data_path, "*_mask.jpg")))[0]
            mask = skimage.io.imread(mask).astype(np.float32) / 255.
            mask = skimage.transform.resize(mask, (512, 512), order=1)
            mask = torch.from_numpy((mask > 0.5).astype(np.float32)).permute(2, 0, 1)
            mask = morph.closing(mask[None, ...], torch.ones(3, 3)).to(global_config.device)
            mask = mask[:, 0, :, :].reshape(-1) > 0

        # get xyz coordinates for each mask pixel
        cam2world_matrix = default_pose[:, :16].view(-1, 4, 4)
        intrinsics = default_pose[:, 16:25].view(-1, 3, 3)
        ray_origins, ray_directions = G.ray_sampler(cam2world_matrix, intrinsics, 512)
        coords = ray_origins + depth * ray_directions

        # project back into canonical pose
        # undo translation
        coords = coords[:, mask, :]
        coords = coords - cam2world_matrix[:, :3, 3]
        # undo rotation
        coords = coords @ cam2world_matrix[:, :3, :3]

    return coords


def get_projected_mask(coords, intrinsics, cam2world_matrix, ksize=15, std=7):
    """ takes 3d point cloud of mask coordinates, projects into a new view, and fills holes """

    # apply rotation
    coords = coords @ cam2world_matrix[:, :3, :3].permute(0, 2, 1)

    # apply translation
    coords = coords + cam2world_matrix[:, :3, 3]

    # project onto camera and then image coordinates
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    x = -fx * coords[..., 0] / (2.7 - coords[..., -1])
    y = -fy * coords[..., 1] / (2.7 - coords[..., -1])

    Nmask = 512
    xc = torch.round((x + cx) * Nmask).long()
    yc = torch.round((y + cy) * Nmask).long()

    invalid = (xc > Nmask-1) | (yc > Nmask-1) | (xc < 0) | (yc < 0)
    xc = xc[~invalid]
    yc = yc[~invalid]

    mask = torch.zeros(Nmask, Nmask, device=global_config.device)
    mask[yc, xc] = 1

    # inpaint convex hull of splatted points
    mask = mask.cpu().numpy()
    mask = torch.from_numpy(convex_hull_image(mask).astype(np.float32)).to(global_config.device)
    mask = mask[None, None, ...]

    # blur things out to feather the compositing
    mask = gaussian_blur2d(mask, (ksize, ksize), (std, std))

    return mask


def composite_results(use_smoothing=False):
    # initial mask projected into 3D point cloud
    init_mask_coords = get_init_mask_coords()
    pose_buff = []

    for idx in tqdm(range(0, 100)):
        # pose for next frame
        target_pose = get_json_pose(args.target_pose_path, f'frames{idx+1:03d}.jpg')

        if use_smoothing:
            if len(pose_buff) < 2:
                pose_buff.append(target_pose)
            else:
                pose_buff[idx % 2] = target_pose

            target_pose = torch.mean(torch.stack(pose_buff, dim=0), dim=0)

        # this is the first frame at the current frame pose
        with torch.no_grad():
            output_bg = G.synthesis(w_full[[0]], target_pose, noise_mode='const', force_fp32=True)
            image_bg = (output_bg['image'] + 1) / 2

        # current frame at its pose
        with torch.no_grad():
            output_fg = G.synthesis(w_full[[idx]], target_pose, noise_mode='const', force_fp32=True)
            image_fg = (output_fg['image'] + 1) / 2

        # get projected_mask
        cam2world_matrix = target_pose[:, :16].view(-1, 4, 4)
        intrinsics = target_pose[:, 16:25].view(-1, 3, 3)
        mask = get_projected_mask(init_mask_coords, intrinsics, cam2world_matrix)

        mask = mask.repeat(1, 3, 1, 1)
        composited = mask * image_fg + (1-mask) * image_bg
        out_img = (np.clip(composited[0].permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

        os.makedirs(f'{args.input_logs_path}/composite', exist_ok=True)
        skimage.io.imsave(f'{args.input_logs_path}/composite/{idx:03d}.png', out_img)


if __name__ == '__main__':
    global out_dir, logdir, G, w_full, scene, args

    args = parser.parse_args()
    out_dir = 'output_2000'
    logdir = os.path.join(args.input_logs_path, out_dir)

    G = load_tuned_G(None, None, full_path=f'{logdir}/model.pt')
    G.sample_mixed = triplane.SRPosedGenerator.sample_mixed.__get__(G)
    G.rendering_kwargs['depth_resolution'] *= 4
    G.rendering_kwargs['depth_resolution_importance'] *= 4
    w_full = torch.from_numpy(np.load(f'{args.input_logs_path}/w.npy')).to(global_config.device)

    composite_results(use_smoothing=True)
