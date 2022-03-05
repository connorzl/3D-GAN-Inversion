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

import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)

    # target reference
    name = testdata[0]['imagename']
    savepath = '{}/{}.jpg'.format(savefolder, name)
    with torch.no_grad():
        images = testdata[0]['image'].to(device)[None,...]
        id_codedict = deca.encode(images)
    
    id_codedict['hr_images'] = testdata[0]['hr_image'].to(device)[None,...]
    id_opdict, id_visdict = deca.decode_coarse(id_codedict, pca_scale=1, all_scale=1)
    id_codedict['attributes'] = id_opdict['attributes']

    all_poses = []
    all_exps = []

    for i in range(0, len(expdata)):
        # source reference
        with torch.no_grad():
            exp_images = expdata[i]['image'].to(device)[None,...]
            exp_codedict = deca.encode(exp_images)
            all_poses.append(exp_codedict['pose'][:,3:])
            all_exps.append(exp_codedict['exp'])
    all_poses = torch.stack(all_poses, 0)
    all_exps = torch.stack(all_exps, 0)

    if args.useSmoothing:
        smooth_poses = []
        smooth_exps = []
        windowsize = 4
        for i in range(0, len(expdata)):
            start = max(0, i-windowsize)
            poses = torch.mean(all_poses[start:start+5], 0)
            exps = torch.mean(all_exps[start:start+5], 0)
            smooth_poses.append(poses)
            smooth_exps.append(exps)
        smooth_poses = torch.stack(smooth_poses, 0)
        smooth_exps = torch.stack(smooth_exps, 0)
    else:
        smooth_poses = all_poses
        smooth_exps = all_exps

    for i in range(0, len(expdata)):
        name = testdata[0]['imagename'] + "_" + str(i).zfill(3)
        print("processing:", i, "/", len(expdata))

        # source reference
        id_codedict['pose'][:,3:] = smooth_poses[i]
        id_codedict['exp'] = smooth_exps[i]

        with torch.no_grad():
            tform = testdata[0]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testdata[0]['original_image'][None, ...].to(device)
            
            if args.scale_expressions:
                orig_opdict, orig_visdict = deca.decode_coarse(id_codedict, render_orig=True, original_image=original_image, tform=tform, pca_index=9, pca_scale=1, all_scale=2, freeze_eyes=id_opdict['freeze_eyes'])
            else:
                orig_opdict, orig_visdict = deca.decode_coarse(id_codedict, render_orig=True, original_image=original_image, tform=tform, pca_index=9, pca_scale=1, all_scale=1, freeze_eyes=id_opdict['freeze_eyes'])

            orig_visdict['inputs'] = original_image  

        depth_image = deca.render.render_depth(orig_opdict['trans_verts']).repeat(1,3,1,1)
        orig_visdict['depth_images'] = depth_image
        cv2.imwrite(os.path.join(savefolder, name + '_depth.jpg'), util.tensor2image(depth_image[0]))

        rendered_image  = util.tensor2image(orig_visdict["rendered_images"][0])
        cv2.imwrite(os.path.join(savefolder, name + '_' + "rendered_images" +'.jpg'), rendered_image)

        mask  = util.tensor2image(orig_visdict["mask"][0])
        cv2.imwrite(os.path.join(savefolder, name + '_' + "mask" +'.jpg'), mask)

        masked_img = copy.deepcopy(rendered_image)
        masked_img[mask[:, :, 0] < 100] = 0
        cv2.imwrite(os.path.join(savefolder, name + '_' + "masked_rendered_images" +'.jpg'), masked_img)
        """
        for j in range(0, 50):
            print("PROCESSING PCA INDEX:", j)
            with torch.no_grad():
                tform = testdata[0]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[0]['original_image'][None, ...].to(device)
                orig_opdict, orig_visdict = deca.decode_coarse(id_codedict, render_orig=True, original_image=original_image, tform=tform, pca_index=j)
                orig_visdict['inputs'] = original_image  

            name = testdata[0]['imagename'] + "_" + str(i).zfill(3) + "_" + str(j).zfill(2)

            depth_image = deca.render.render_depth(orig_opdict['trans_verts']).repeat(1,3,1,1)
            orig_visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name + '_depth.jpg'), util.tensor2image(depth_image[0]))

            rendered_image  = util.tensor2image(orig_visdict["rendered_images"][0])
            cv2.imwrite(os.path.join(savefolder, name + '_' + "rendered_images" +'.jpg'), rendered_image)

            mask  = util.tensor2image(orig_visdict["mask"][0])
            cv2.imwrite(os.path.join(savefolder, name + '_' + "mask" +'.jpg'), mask)

            masked_img = copy.deepcopy(rendered_image)
            masked_img[mask[:, :, 0] < 100] = 0
            cv2.imwrite(os.path.join(savefolder, name + '_' + "masked_rendered_images" +'.jpg'), masked_img)

        #masked_img = torch.permute(torch.from_numpy(np.expand_dims(masked_img, 0)), (0, 3, 1, 2)) / 255
        #orig_visdict['masked_rendered_images'] = masked_img[0]
        #cv2.imwrite(os.path.join(savefolder, name + '_edited_target.jpg'), deca.visualize(orig_visdict))
        """
        
    print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/7.jpg', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )

    parser.add_argument('--useSmoothing', action='store_true',
                        help='whether to smooth')

    parser.add_argument('--scale_expressions', action='store_true',
                        help='whether to scale expressions up')

    main(parser.parse_args())
