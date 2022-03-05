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
    i = 0
    name = testdata[i]['imagename']
    savepath = '{}/{}.jpg'.format(savefolder, name)
    images = testdata[i]['image'].to(device)[None,...]
    hr_images = testdata[i]['hr_image'].to(device)[None,...]
    with torch.no_grad():
        id_codedict = deca.encode(images)
    id_codedict['hr_images'] = hr_images
    id_opdict, id_visdict = deca.decode(id_codedict)
    id_codedict['attributes'] = id_opdict['attributes']
    id_codedict['dense_attributes'] = id_opdict['dense_attributes']
    
    # source reference
    exp_images = expdata[i]['image'].to(device)[None,...]
    exp_images_hr = expdata[i]['hr_image'].to(device)[None,...]
    with torch.no_grad():
        exp_codedict = deca.encode(exp_images)
    exp_codedict['hr_images'] = exp_images_hr
    exp_opdict, exp_visdict = deca.decode(exp_codedict)

    # transfer exp code
    id_codedict['pose'][:,3:] = exp_codedict['pose'][:,3:]
    id_codedict['exp'] = exp_codedict['exp']
    transfer_opdict, transfer_visdict = deca.decode(id_codedict)
    
    tform = testdata[i]['tform'][None, ...]
    tform = torch.inverse(tform).transpose(1,2).to(device)
    original_image = testdata[i]['original_image'][None, ...].to(device)
    _, orig_visdict = deca.decode(id_codedict, render_orig=True, original_image=original_image, tform=tform)
    orig_visdict['inputs'] = original_image  
    cv2.imwrite(os.path.join(savefolder, name + '_uncroppped_edited_target.jpg'), deca.visualize(orig_visdict))

    image  = util.tensor2image(orig_visdict["rendered_images_detailed"][0])
    cv2.imwrite(os.path.join(savefolder, name + '_' + "rendered_images_detailed" +'.jpg'), image)

    image  = util.tensor2image(orig_visdict["mask_detailed"][0])
    cv2.imwrite(os.path.join(savefolder, name + '_' + "mask_detailed" +'.jpg'), image)

    image  = util.tensor2image(orig_visdict["rendered_images"][0])
    cv2.imwrite(os.path.join(savefolder, name + '_' + "rendered_images" +'.jpg'), image)

    image  = util.tensor2image(orig_visdict["mask"][0])
    cv2.imwrite(os.path.join(savefolder, name + '_' + "mask" +'.jpg'), image)


    # -- save results
    transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']
    if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
        os.makedirs(os.path.join(savefolder, name, 'original_target'), exist_ok=True)
        os.makedirs(os.path.join(savefolder, name, 'edited_target'), exist_ok=True)

    image_name = name
    for save_type in ['original_target', 'edited_target']:
        if save_type == 'reconstruction':
            visdict = id_visdict; opdict = id_opdict
        else:
            visdict = transfer_visdict; opdict = transfer_opdict
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, save_type, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, save_type, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, save_type, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, save_type, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, save_type, name + '.mat'), opdict)
        if args.saveImages:
            for vis_name in ['inputs', 'hr_inputs', 'rendered_images', 'rendered_images_detailed', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image  = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, save_type, name + '_' + vis_name +'.jpg'), image)
    cv2.imwrite(os.path.join(savefolder, name + '_original_target.jpg'), deca.visualize(id_visdict))
    cv2.imwrite(os.path.join(savefolder, name + '_original_source.jpg'), deca.visualize(exp_visdict))
    cv2.imwrite(os.path.join(savefolder, name + '_edited_target.jpg'), deca.visualize(transfer_visdict))
    cv2.imwrite(os.path.join(savefolder, name + '_original_edited_target.jpg'), deca.visualize(orig_visdict))
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
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())

    main(parser.parse_args())
