import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
from functools import partial
import time

def model_fn_decorator(loss_fn, device, mode='train'):
    def _gaussian_weights(tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=device), (nbatches, 64, 1, 1))
    
    def tile_forward(model, high, low, h_ori, w_ori, x, y):
        b, c, h, w = high.size()

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/16)*16 - w) // 2
        h_pad = (math.ceil(h/16)*16 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        if w_pad != 0 or h_pad != 0:
            high = img_pad(high, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            st = time.time()
            out = model(high, low, h_ori, w_ori, x, y, mode="test")
            cur_time = time.time()-st

            if h_pad != 0:
                out = out[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                out = out[:, :, :, w_pad:-w_odd_pad]
            
        return out, cur_time
    
    def actual_tile_forward(model, high, low, h_ori, w_ori, x, y):
        b, c, h, w = high.size()

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/16)*16 - w) // 2
        h_pad = (math.ceil(h/16)*16 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        high = img_pad(high, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        # print(w_pad, h_pad, w_odd_pad, h_odd_pad) # 0 4 0 4
        # assert 0
        # low = img_pad(low, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            st = time.time()
            h1, h2, h3, h_f = model.model_high(high)
            
            
            return h_f, 0
            
            out = model(high, low, h_ori, w_ori, x, y, mode="test")
            cur_time = time.time()-st
            assert out.shape[2] == 2160
            assert out.shape[3] == 3840
            assert h_pad == 0 and w_pad == 0
            # if h_pad != 0:
            #     out = out[:, :, h_pad:-h_odd_pad, :]
            # if w_pad != 0:
            #     out = out[:, :, :, w_pad:-w_odd_pad]
            
        return out, cur_time
    
    @torch.no_grad()
    def test_model_fn(args, data, model, save_path, compute_metrics):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0

        high = data['crop_moire_high'].to(device)
        low = data['resized_morie_low'].to(device)
        
        h_ori = data['h_ori'].to(device)
        w_ori = data['w_ori'].to(device)
        x = data['x'].to(device)
        y = data['y'].to(device)
        label = data['label'].to(device)
        
        cur_time = 0
        out_1 = None
        
        if not args.TILE:
            out_1, cur_time = tile_forward(model, high, low, h_ori, w_ori, x, y)
        else:
            b, c, h, w = high.size()
            assert b == 1
            tile_size = args.TILE_SIZE
            tile_overlap = args.TILE_OVERLAP
            if h * w <= tile_size * tile_size:
                # print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
                out_1, cur_time = tile_forward(model, high, low, h_ori, w_ori, x, y)
            else:
                # print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
                tile_weights = _gaussian_weights(tile_size, tile_size, 1)
                tile_size = min(tile_size, min(h, w))
                tile_weights = _gaussian_weights(tile_size, tile_size, 1)

                grid_rows = 0
                cur_x = 0
                while cur_x < high.size(-1) // 2:
                    cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                    grid_rows += 1

                grid_cols = 0
                cur_y = 0
                while cur_y < high.size(-2) // 2:
                    cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                    grid_cols += 1

                input_high_list = []
                input_low_list = []
                noise_preds = []
                for row in range(grid_rows):
                    noise_preds_row = []
                    for col in range(grid_cols):
                        if col < grid_cols-1 or row < grid_rows-1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size * 2-tile_overlap * row, 0)
                            ofs_y = max(col * tile_size * 2-tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows-1:
                            ofs_x = w - tile_size * 2
                        if col == grid_cols-1:
                            ofs_y = h - tile_size * 2

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size * 2
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size * 2

                        # input tile dimensions
                        input_high_tile = high[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                        input_high_list.append(input_high_tile)
                        
                        input_low_tile = low[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                        input_low_list.append(input_low_tile)

                        if len(input_high_list) == 1 or col == grid_cols-1:
                            input_high_list_t = torch.cat(input_high_list, dim=0)
                            input_low_list_t = torch.cat(input_low_list, dim=0)
                            # predict the noise residual
                            out_1, tile_time = actual_tile_forward(model, input_high_list_t, input_low_list_t, h_ori, w_ori, x, y)
                            cur_time += tile_time
                            input_high_list = []
                            input_low_list = []
                            # assert out_1.shape[2] == 512, f"{out_1.shape}"
                        noise_preds.append(out_1)

                # Stitch noise predictions for all tiles
                noise_pred = torch.zeros([b, 64, h // 2, w // 2], device=device)
                contributors = torch.zeros([b, 64, h // 2, w // 2], device=device)
                # Add each tile contribution to overall latents
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if col < grid_cols-1 or row < grid_rows-1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size-tile_overlap * row, 0)
                            ofs_y = max(col * tile_size-tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows-1:
                            ofs_x = w // 2 - tile_size
                        if col == grid_cols-1:
                            ofs_y = h // 2 - tile_size

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size

                        noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                        contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                        
                # exit(0)
                # Average overlapping areas with more than 1 contributor
                noise_pred /= contributors
                h_f = noise_pred
                
                l1, l2, l3, l_f = model.model_low(low)
                
                l_f = F.interpolate(l_f, size=(h_f.shape[2], h_f.shape[3]), mode='bicubic',)
                
                torch.cuda.empty_cache()
                del l1, l2, l3 
                del contributors
                # with torch.no_grad():
                    # out_1 = model.in_forward(h_f, l_f)
                fusion = model.fusion_pre_high(h_f) + model.fusion_pre_low(l_f)
                # assert 0, f"{fusion.shape}"
                res = model.fusion_module(fusion)
                res = model.out_conv(res)
                res = F.pixel_shuffle(res, 2)
                out_1 = res

        if args.EVALUATION_METRIC:
            cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(out_1, label)

        # save images
        if args.SAVE_IMG:
            out_save = out_1.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG)

        return cur_psnr, cur_ssim, cur_lpips, cur_time

    def model_fn(args, data, model, iters):
        model.train()
        # prepare input and forward
        # in_img = data['in_img'].to(device)
        high = data['crop_moire_high'].to(device)
        low = data['resized_morie_low'].to(device)
        
        h_ori = data['h_ori'].to(device)
        w_ori = data['w_ori'].to(device)
        x = data['x'].to(device)
        y = data['y'].to(device)
        
        label = data['label'].to(device)
        out_1, out_2, out_3 = model(high, low, h_ori, w_ori, x, y)
        loss = loss_fn(out_1, out_2, out_3, label)

        # save images
        if iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
            # in_save = in_img.detach().cpu()
            out_save = out_1.detach().cpu()
            gt_save = label.detach().cpu()
            res_save = torch.cat((out_save, gt_save), 3)
            save_number = (iters + 1) // args.SAVE_ITER
            torchvision.utils.save_image(res_save,
                                         args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg')


        return loss

    if mode == 'test':
        fn = test_model_fn
    else:
        fn = model_fn
    return fn