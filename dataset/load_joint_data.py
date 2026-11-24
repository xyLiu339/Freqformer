import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile
import os
from torchvision.transforms import ToTensor, ToPILImage
from dataset.color_fix import split_to_high_low
import torch.nn.functional as F

def create_dataset(
    args,
    data_path,
    mode='train',
):
    def _list_image_files_recursively(data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('gt.jpg'):
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list

    if args.DATA_TYPE == 'UHDM':
        uhdm_files = _list_image_files_recursively(data_dir=data_path)
        dataset = uhdm_data_loader(args, uhdm_files, mode=mode)
    elif args.DATA_TYPE == 'FHDMi':
        fhdmi_files = sorted([file for file in os.listdir(data_path + '/target') if file.endswith('.png')])
        dataset = fhdmi_data_loader(args, fhdmi_files, mode=mode)
    elif args.DATA_TYPE == 'TIP':
        tip_files = sorted([file for file in os.listdir(data_path + '/source') if file.endswith('.png')])
        dataset = tip_data_loader(args, tip_files, mode=mode)
    elif args.DATA_TYPE == 'AIM':
        if mode=='train':
            aim_files = sorted([file for file in os.listdir(data_path + '/moire') if file.endswith('.jpg')])
        else:
            aim_files = sorted([file for file in os.listdir(data_path + '/moire') if file.endswith('.png')])
        dataset = aim_data_loader(args, aim_files, mode=mode)
    else:
        print('Unrecognized data_type!')
        raise NotImplementedError
    data_loader = data.DataLoader(
        dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKER, drop_last=True
    )

    return data_loader


class uhdm_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_tar = self.image_list[index]
        number = os.path.split(path_tar)[-1][0:4]
        path_src = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_moire.jpg'
        if self.mode == 'train':
            if self.loader == 'crop':
                if os.path.split(path_tar)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                x = random.randint(0, w - self.args.CROP_SIZE)
                y = random.randint(0, h - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
                
            elif self.loader == 'joint':
                if os.path.split(path_tar)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                x = int(random.randint(0, w - self.args.CROP_SIZE-32) // 4 * 4)
                y = int(random.randint(0, h - self.args.CROP_SIZE-32) // 4 * 4)
                assert y+self.args.CROP_SIZE <= h and x+self.args.CROP_SIZE <= w, f"{x, y, self.args.CROP_SIZE, h, w}"
                crop_moire_high, resized_morie_low, crop_gt_high, resized_gt_low, gt_img = \
                    joint_loader(self.args.RESIZE_SIZE,  self.args.CROP_SIZE, x, y, path_tar, path_src)

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            elif self.loader == 'joint':
                h = 2160
                w = 3840
                x = int(random.randint(0, 1920 - self.args.CROP_SIZE) // 4 * 4)
                y = int(random.randint(0, 1080 - self.args.CROP_SIZE) // 4 * 4)
                crop_moire_high, resized_morie_low, crop_gt_high, resized_gt_low, gt_img = \
                    joint_test_loader(self.args.RESIZE_SIZE,  self.args.CROP_SIZE, x, y, path_tar, path_src)
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        # data['in_img'] = moire_imgs
        # data['label'] = labels
        data['number'] = number
        
        data['crop_moire_high'] = crop_moire_high
        data['resized_morie_low'] = resized_morie_low
        data['crop_gt_high'] = crop_gt_high
        data['resized_gt_low'] = resized_gt_low
        
        data['h_ori'] = h
        data['w_ori'] = w
        data['x'] = x
        data['y'] = y
        
        data['label'] = gt_img

        return data

    def __len__(self):
        return len(self.image_list)


class fhdmi_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = image_in_gt[4:9]
        image_in = 'src_' + number + '.png'
        if self.mode == 'train':
            path_tar = self.args.TRAIN_DATASET + '/target/' + image_in_gt
            path_src = self.args.TRAIN_DATASET + '/source/' + image_in
            if self.loader == 'crop':
                x = random.randint(0, 1920 - self.args.CROP_SIZE)
                y = random.randint(0, 1080 - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
                
            elif self.loader == 'joint':
                x = int(random.randint(0, 1920 - self.args.CROP_SIZE) // 4 * 4)
                y = int(random.randint(0, 1080 - self.args.CROP_SIZE) // 4 * 4)
                crop_moire_high, resized_morie_low, crop_gt_high, resized_gt_low, gt_img = \
                    joint_loader(self.args.RESIZE_SIZE,  self.args.CROP_SIZE, x, y, path_tar, path_src)

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            path_tar = self.args.TEST_DATASET + '/target/' + image_in_gt
            path_src = self.args.TEST_DATASET + '/source/' + image_in
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            elif self.loader == 'joint':
                x = int(random.randint(0, 1920 - self.args.CROP_SIZE) // 4 * 4)
                y = int(random.randint(0, 1080 - self.args.CROP_SIZE) // 4 * 4)
                crop_moire_high, resized_morie_low, crop_gt_high, resized_gt_low, gt_img = \
                    joint_test_loader(self.args.RESIZE_SIZE,  self.args.CROP_SIZE, x, y, path_tar, path_src)

            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        # data['in_img'] = moire_imgs
        # data['label'] = labels
        data['number'] = number
        
        data['crop_moire_high'] = crop_moire_high
        data['resized_morie_low'] = resized_morie_low
        data['crop_gt_high'] = crop_gt_high
        data['resized_gt_low'] = resized_gt_low
        
        data['h_ori'] = 1080
        data['w_ori'] = 1920
        data['x'] = x
        data['y'] = y
        
        data['label'] = gt_img

        return data

    def __len__(self):
        return len(self.image_list)


class aim_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = image_in_gt[0:6]
        image_in = number + '.jpg'
        if self.mode == 'train':
            path_tar = self.args.TRAIN_DATASET + '/clear/' + image_in_gt
            path_src = self.args.TRAIN_DATASET + '/moire/' + image_in
            if self.loader == 'crop':
                x = random.randint(0, 1024 - self.args.CROP_SIZE)
                y = random.randint(0, 1024 - self.args.CROP_SIZE)
                labels, moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_tar, path_src])
                
            elif self.loader == 'joint':
                x = int(random.randint(0, 1024 - self.args.CROP_SIZE) // 4 * 4)
                y = int(random.randint(0, 1024 - self.args.CROP_SIZE) // 4 * 4)
                crop_moire_high, resized_morie_low, crop_gt_high, resized_gt_low, gt_img = \
                    joint_loader(self.args.RESIZE_SIZE,  self.args.CROP_SIZE, x, y, path_tar, path_src)

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            image_in = number + '.png'
            image_in_gt = number + '.png'
            path_tar = self.args.TEST_DATASET + '/clear/' + image_in_gt
            path_src = self.args.TEST_DATASET + '/moire/' + image_in
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args.RESIZE_SIZE, [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            elif self.loader == 'joint':
                x = int(random.randint(0, 1024 - self.args.CROP_SIZE) // 4 * 4)
                y = int(random.randint(0, 1024 - self.args.CROP_SIZE) // 4 * 4)
                crop_moire_high, resized_morie_low, crop_gt_high, resized_gt_low, gt_img = \
                    joint_test_loader(self.args.RESIZE_SIZE,  self.args.CROP_SIZE, x, y, path_tar, path_src)
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        # data['in_img'] = moire_imgs
        # data['label'] = labels
        data['number'] = number
        
        data['crop_moire_high'] = crop_moire_high
        data['resized_morie_low'] = resized_morie_low
        data['crop_gt_high'] = crop_gt_high
        data['resized_gt_low'] = resized_gt_low
        
        data['h_ori'] = 1024
        data['w_ori'] = 1024
        data['x'] = x
        data['y'] = y
        
        data['label'] = gt_img

        return data

    def __len__(self):
        return len(self.image_list)


class tip_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        t_list = [transforms.ToTensor()]
        self.composed_transform = transforms.Compose(t_list)

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in = self.image_list[index]
        image_in_gt = image_in[:-10] + 'target.png'
        number = image_in_gt[:-11]

        if self.mode == 'train':
            labels = self.default_loader(self.args.TRAIN_DATASET + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args.TRAIN_DATASET + '/source/' + image_in)
            
            moire_high_ori, moire_low_ori = split_to_high_low(moire_imgs) # C * H * W
            gt_high_ori, gt_low_ori = split_to_high_low(labels) # C * H * W

            w, h = labels.size
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            labels = labels.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            # moire_imgs = moire_imgs.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            # crop the tensor
            moire_high = moire_high_ori[:, int(h / 6) + j:int(h * 5 / 6) + j, int(w / 6) + i:int(w * 5 / 6) + i]
            gt_high = gt_high_ori[:, int(h / 6) + j:int(h * 5 / 6) + j, int(w / 6) + i:int(w * 5 / 6) + i]
            moire_low = moire_low_ori[:, int(h / 6) + j:int(h * 5 / 6) + j, int(w / 6) + i:int(w * 5 / 6) + i]
            gt_low = gt_low_ori[:, int(h / 6) + j:int(h * 5 / 6) + j, int(w / 6) + i:int(w * 5 / 6) + i]
            
            labels = labels.resize((256, 256), Image.BILINEAR)
            labels = self.composed_transform(labels)
            
            if moire_high.shape[1] <= 1 or moire_high.shape[2] <= 1:
                print('Warning: image size is too small, set to 870')
                print("The path is: ", image_in_gt, i, j)
                moire_high = moire_high_ori
                gt_high = gt_high_ori
                moire_low = moire_low_ori
                gt_low = gt_low_ori
            
            crop_moire_high = F.interpolate(
                moire_high.unsqueeze(0), 
                size=(256,256), 
                mode='bicubic',
            )[0].cpu()
            
            crop_gt_high = F.interpolate(
                gt_high.unsqueeze(0), 
                size=(256,256), 
                mode='bicubic',
            )[0].cpu()
            
            resized_morie_low = F.interpolate(
                moire_low.unsqueeze(0), 
                size=(256, 256), 
                mode='bicubic',
            )[0].cpu()
            
            resized_gt_low = F.interpolate(
                gt_low.unsqueeze(0), 
                size=(256, 256), 
                mode='bicubic',
            )[0].cpu()
                        
            x = 0
            y = 0

        elif self.mode == 'test':
            labels = self.default_loader(self.args.TEST_DATASET + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args.TEST_DATASET + '/source/' + image_in)
            
            moire_high, moire_low = split_to_high_low(moire_imgs) # C * H * W
            gt_high, gt_low = split_to_high_low(labels) # C * H * W

            w, h = labels.size
            labels = labels.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            # moire_imgs = moire_imgs.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            # crop the tensor
            moire_high = moire_high[:, int(h / 6):int(h * 5 / 6), int(w / 6):int(w * 5 / 6)]
            gt_high = gt_high[:, int(h / 6):int(h * 5 / 6), int(w / 6):int(w * 5 / 6)]
            moire_low = moire_low[:, int(h / 6):int(h * 5 / 6), int(w / 6):int(w * 5 / 6)]
            gt_low = gt_low[:, int(h / 6):int(h * 5 / 6), int(w / 6):int(w * 5 / 6)]
            
            labels = labels.resize((256, 256), Image.BILINEAR)
            labels = self.composed_transform(labels)
            
            crop_moire_high = F.interpolate(
                moire_high.unsqueeze(0), 
                size=(256,256), 
                mode='bicubic',
            )[0].cpu()
            
            crop_gt_high = F.interpolate(
                gt_high.unsqueeze(0), 
                size=(256,256), 
                mode='bicubic',
            )[0].cpu()
            
            resized_morie_low = F.interpolate(
                moire_low.unsqueeze(0), 
                size=(256, 256), 
                mode='bicubic',
            )[0].cpu()
            
            resized_gt_low = F.interpolate(
                gt_low.unsqueeze(0), 
                size=(256, 256), 
                mode='bicubic',
            )[0].cpu()
                        
            x = 0
            y = 0

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        # moire_imgs = self.composed_transform(moire_imgs)
        # labels = self.composed_transform(labels)

        # data['in_img'] = moire_imgs
        # data['label'] = labels
        data['number'] = number
        
        data['crop_moire_high'] = crop_moire_high
        data['resized_morie_low'] = resized_morie_low
        data['crop_gt_high'] = crop_gt_high
        data['resized_gt_low'] = resized_gt_low
        
        data['h_ori'] = 256
        data['w_ori'] = 256
        data['x'] = x
        data['y'] = y
        
        data['label'] = labels

        return data

    def __len__(self):
        return len(self.image_list)



    
def default_loader(path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs
    
def crop_loader(crop_size, x, y, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.crop((x, y, x + crop_size, y + crop_size))
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs


def tip_joint_test_loader(resize_size, crop_size, x, y, path_tar, path_src):
    # process the moire_img    
    moire_img = Image.open(path_src).convert('RGB')
    
    moire_high, moire_low = split_to_high_low(moire_img) # C * H * W
   
    # crop_moire_high = moire_high[:, y:y + crop_size, x:x + crop_size]
    assert (moire_low < 1.001).all() and (moire_low > -0.001).all(), 'low frequency should be in [0,1]'
    
    resized_morie_low = F.interpolate(
        moire_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    # process the gt_img
    gt_img = Image.open(path_tar).convert('RGB')
    
    gt_high, gt_low = split_to_high_low(gt_img) # C * H * W
    
    # crop_gt_high = gt_high[:, y:y + crop_size, x:x + crop_size]
    assert (gt_low < 1.001).all() and (gt_low > -0.001).all(), 'low frequency should be in [0,1]'
    resized_gt_low = F.interpolate(
        gt_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    gt_img = Image.open(path_tar).convert('RGB')
    # gt_img = gt_img.crop((x, y, x + crop_size, y + crop_size))
    gt_img = default_toTensor(gt_img)
    
    return moire_high.cpu(), resized_morie_low.cpu(), gt_high.cpu(), resized_gt_low.cpu(), gt_img

def tip_joint_loader(resize_size, crop_size, x, y, path_tar, path_src):
    # process the moire_img    
    moire_img = Image.open(path_src).convert('RGB')
    
    moire_high, moire_low = split_to_high_low(moire_img) # C * H * W
   
    crop_moire_high = moire_high[:, y:y + crop_size, x:x + crop_size]
    assert (moire_low < 1.001).all() and (moire_low > -0.001).all(), 'low frequency should be in [0,1]'
    
    resized_morie_low = F.interpolate(
        moire_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    # process the gt_img
    gt_img = Image.open(path_tar).convert('RGB')
    
    gt_high, gt_low = split_to_high_low(gt_img) # C * H * W
    
    crop_gt_high = gt_high[:, y:y + crop_size, x:x + crop_size]
    assert (gt_low < 1.001).all() and (gt_low > -0.001).all(), 'low frequency should be in [0,1]'
    resized_gt_low = F.interpolate(
        gt_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    gt_img = Image.open(path_tar).convert('RGB')
    gt_img = gt_img.crop((x, y, x + crop_size, y + crop_size))
    gt_img = default_toTensor(gt_img)
    
    return crop_moire_high.cpu(), resized_morie_low.cpu(), crop_gt_high.cpu(), resized_gt_low.cpu(), gt_img


def joint_test_loader(resize_size, crop_size, x, y, path_tar, path_src):
    # process the moire_img    
    moire_img = Image.open(path_src).convert('RGB')
    
    moire_high, moire_low = split_to_high_low(moire_img) # C * H * W
   
    # crop_moire_high = moire_high[:, y:y + crop_size, x:x + crop_size]
    assert (moire_low < 1.001).all() and (moire_low > -0.001).all(), 'low frequency should be in [0,1]'
    
    resized_morie_low = F.interpolate(
        moire_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    # process the gt_img
    gt_img = Image.open(path_tar).convert('RGB')
    
    gt_high, gt_low = split_to_high_low(gt_img) # C * H * W
    
    # crop_gt_high = gt_high[:, y:y + crop_size, x:x + crop_size]
    assert (gt_low < 1.001).all() and (gt_low > -0.001).all(), 'low frequency should be in [0,1]'
    resized_gt_low = F.interpolate(
        gt_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    gt_img = Image.open(path_tar).convert('RGB')
    # gt_img = gt_img.crop((x, y, x + crop_size, y + crop_size))
    gt_img = default_toTensor(gt_img)
    
    return moire_high.cpu(), resized_morie_low.cpu(), gt_high.cpu(), resized_gt_low.cpu(), gt_img

def joint_loader(resize_size, crop_size, x, y, path_tar, path_src):
    # process the moire_img    
    moire_img = Image.open(path_src).convert('RGB')
    
    moire_high, moire_low = split_to_high_low(moire_img) # C * H * W
   
    crop_moire_high = moire_high[:, y:y + crop_size, x:x + crop_size]
    assert (moire_low < 1.001).all() and (moire_low > -0.001).all(), 'low frequency should be in [0,1]'
    
    resized_morie_low = F.interpolate(
        moire_low.unsqueeze(0), 
        size=(resize_size, resize_size), 
        mode='bicubic',
    )[0]
    
    # process the gt_img
    # gt_img = Image.open(path_tar).convert('RGB')
    
    # gt_high, gt_low = split_to_high_low(gt_img) # C * H * W
    
    # crop_gt_high = gt_high[:, y:y + crop_size, x:x + crop_size]
    # assert (gt_low < 1.001).all() and (gt_low > -0.001).all(), 'low frequency should be in [0,1]'
    # resized_gt_low = F.interpolate(
    #     gt_low.unsqueeze(0), 
    #     size=(resize_size, resize_size), 
    #     mode='bicubic',
    # )[0]
    
    gt_img = Image.open(path_tar).convert('RGB')
    gt_img = gt_img.crop((x, y, x + crop_size, y + crop_size))
    gt_img = default_toTensor(gt_img)
    
    return crop_moire_high.cpu(), resized_morie_low.cpu(), gt_img.cpu(), gt_img.cpu(), gt_img

    
def resize_loader(resize_size, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.resize((resize_size,resize_size),Image.LANCZOS)
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)

