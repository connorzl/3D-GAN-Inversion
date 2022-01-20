# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import sys
import argparse
import torch
from glob import glob
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
import cv2
from decalib.utils import util


def main(args):
    device = args.device

    folders = sorted(glob(os.path.join(args.dataset_dir, '*')))

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    for folder in tqdm(folders):

        # load test images
        if args.subfolder is not None:
            folder = os.path.join(folder, args.subfolder)

        print(folder)
        testdata = datasets.TestData(folder, iscrop=args.iscrop, face_detector=args.detector)

        # target reference
        freeze_eyes = None
        for idx, sample in enumerate(testdata):
            images = sample['image'].to(device)[None, ...]
            with torch.no_grad():
                code = deca.encode(images)

                os.makedirs(os.path.join(folder, 'detections'), exist_ok=True)
                out = np.concatenate((code['exp'].cpu().numpy(), code['pose'].cpu().numpy()), axis=1)
                np.savetxt(os.path.join(folder, 'detections', f'frames{idx:03d}_deca.txt'), out)

                # freeze vertices to be the same as the first frame,
                # then only solve for the updated expression vectors
                # rather than allowing the shape to change
                new_expr, freeze_eyes = deca.flame.project_expr(shape_params=code['shape'],
                                                                expression_params=code['exp'],
                                                                pose_params=code['pose'],
                                                                pca_index=0, pca_scale=1,
                                                                all_scale=1,
                                                                freeze_eyes=freeze_eyes)

                out[:, :50] = new_expr.cpu().numpy()
                np.savetxt(os.path.join(folder, 'detections', f'frames{idx:03d}_deca_noeyes.txt'), out)

                # hr_images = sample['hr_image'].to(device)[None, ...]
                # print(hr_images.shape)
                # code['hr_images'] = hr_images
                # opdict, visdict = deca.decode(code)
                # image = util.tensor2image(visdict['landmarks2d'][0])
                # cv2.imwrite(os.path.join('test.jpg'), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('--dataset_dir', default='/home/lindell/workspace/PIRender/target', type=str,
                        help='set device, cpu for using cpu')

    parser.add_argument('--subfolder', default=None, type=str,
                        help='set device, cpu for using cpu')

    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')

    main(parser.parse_args())
