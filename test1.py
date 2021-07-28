"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from os import mkdir
from os.path import isfile, isdir
from collections import OrderedDict
import torch
from torch.autograd import Variable
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
import pandas as pd

from options.test_options import TestOptions
from deepsee_models.sr_model import SRModel
from util.visualizer import Visualizer
import util.visualizer
from measure import lpips_analysis
import os


def run(opt):
    # load model
    model_dir = opt.model_dir
    model_name = opt.which_epoch + "_net_SR.pth"
    model_path = os.path.join(model_dir, model_name)
    model = SRModel(opt)
    device=torch.device('cuda:0')
    model = model.to(device)
    #model.load_state_dict(torch.load(model_path).state_dict(), strict=True)
    model.eval()

    measure_results = []

    visualizer = Visualizer(opt)

    with torch.no_grad():

        # Input images
        test_image_dir = opt.test_lr_dir
        img_path = os.path.join(test_image_dir, "*.png")
        files = glob.glob(img_path)
        files.sort()

        for fn in tqdm(files):
            # Load images
            img = Image.open(fn)
            img = np.asarray(img).astype(np.float32) / 255.0 # HxWxC
            img_lr = np.transpose(img, [2, 0, 1]) # CxHxW
            img_lr = img_lr[np.newaxis, ...] # BxCxHxW
            img_lr = Variable(torch.from_numpy(img_lr), volatile=True).to(device)
            img_sr = []

            _, _, H, W = img_lr.size()
            dh, dw = 4, 4
            i_pad = 10

            # Generate SR
            for i in range(opt.SR_num):
                '''
                img_lr = img_lr[:,:,:128,:128]
                _,_,HH,WW=img_lr.size()
                noise = torch.randn(1, 128, HH, WW)
                img_hr = model.generate_fake(img_lr, noise)
                print('finish {}'.format(i))
                '''
                hs = []
                for h in range(dh):
                    sh = (H//dh)*h - i_pad
                    eh = (H//dh)*(h+1) + i_pad
                    hh_pad = [i_pad*4, i_pad*4]
                    if sh < 0:
                        sh = 0
                        hh_pad[0] = 0
                    if eh > H:
                        eh = H
                        hh_pad[1] = 0
                    #print('hh_pad:',hh_pad)    #[0,8]
                    
                    ws = []
                    for w in range(dw):
                        sw = (W//dw)*w - i_pad
                        ew = (W//dw)*(w+1) + i_pad
                        ww_pad = [i_pad*4, i_pad*4]
                        if sw < 0:
                            sw = 0
                            ww_pad[0] = 0
                        if ew > W:
                            es = W
                            ww_pad[1] = 0
                        #print('ww_pad:',ww_pad)    #[0,8]
                        
                        slice_input = img_lr[:,:,sh:eh,sw:ew]
                        
                        if opt.with_noise:
                            _, _, HH, WW = slice_input.size()
                            noise = torch.randn(1, 128, HH, WW)
                            #print('noise:',noise.size())       #[194,257]
                            slice_output = model.generate_fake(slice_input, noise)
                        else:
                            slice_output = model.generate_fake(slice_input)

                        #print('slice_outpt:',slice_output.size())
                        slice_output = (slice_output).cpu().data.numpy()
                        slice_output = np.clip(slice_output[0], 0., 1.)
                        slice_output = np.transpose(slice_output, [1,2,0])
                        
                        if hh_pad[0] > 0:
                            slice_output = slice_output[hh_pad[0]:,:,:]
                        if hh_pad[1] > 0:
                            slice_output = slice_output[:-hh_pad[1],:,:]
                        if ww_pad[0] > 0:
                            slice_output = slice_output[:,ww_pad[0]:,:]
                        if ww_pad[1] > 0:
                            slice_output = slice_output[:,:-ww_pad[1],:]
                            
                        ws.append(slice_output)
                        
                    hs.append(np.concatenate(ws,1))
                img_hr = np.concatenate(hs,0)
                
                # Save SR
                save_dir = opt.results_dir
                if not isdir(save_dir):
                    mkdir(save_dir)
                save_name = '{}_'.format(fn[-8:-4]) + str(i) + '.png'
                save_path = os.path.join(save_dir, save_name)
                Image.fromarray(np.around(img_hr*255).astype(np.uint8)).save(save_path)

                img_sr.append(img_hr)
            
            # Measure
            if(not opt.only_generate_SR):         
                # Load GT
                fn_name = fn[-8:]  # to find corresponding GT
                img_gt_m = opt.groundtruth_dir + fn_name
                # Load SR
                img_sr_m = []
                for i in range(opt.SR_num):
                    img_sr_m.append(save_dir+'{}_'.format(fn[-8:-4]) + str(i) + '.png')
                measure_results.append(lpips_analysis(img_gt_m, img_sr_m, opt.scale)) # input img name but not tensor

        if(not opt.only_generate_SR):
            # Save measure results
            df = pd.DataFrame(measure_results)
            df_mean = df.mean()
            measure_results_name = opt.measure_results_name
            measure_results_path = opt.measure_results_path
            df.to_csv(measure_results_path+f"{measure_results_name}.csv")
            df_mean.to_csv(measure_results_path+f"{measure_results_name}_mean.csv")



if __name__ == "__main__":
    # Parse arguments
    opt = TestOptions().parse()
    run(opt)




