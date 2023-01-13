import os
import cv2
import PIL
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from models.functions import adaptive_bins, get_bins_masks
#several data augumentation strategies
def cv_random_flip(img, label,depth,bins_depth,gt_stack):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        bins_depth = bins_depth.transpose(Image.FLIP_LEFT_RIGHT)
        gt_stack = gt_stack.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth, bins_depth,gt_stack
def randomCrop(image, label,depth, bins_depth,gt_stack):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region), bins_depth.crop(random_region), gt_stack.crop(random_region)
def randomRotation(image,label,depth,bins_depth,gt_stack):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
        bins_depth = bins_depth.rotate(random_angle, mode)
        gt_stack = gt_stack.rotate(random_angle, mode)
    return image,label,depth,bins_depth,gt_stack
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def bin_trans(bin_masks, depth_transs):
    depth_trans = depth_transs.astype(np.float64) / 255.0
    depth_trans = torch.from_numpy(depth_trans).float()

    bins_mask = torch.from_numpy(bin_masks).float()
    h, w = depth_trans.size()
    bins_depth = depth_trans.view(1, h, w).repeat(3, 1, 1)
    bins_depth = bins_depth * bins_mask
    for i in range(3):
        bins_depth[i] = bins_depth[i] / bins_depth[i].max()
    return bins_depth
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

def get_gt_stack(gt):
    mask1 = (gt == 4)
    mask2 = (gt == 3)
    mask3 = (gt == 2)
    mask4 = (gt == 1)
    height, wide = gt.shape
    temp_file1 = np.zeros((height,wide))
    temp_file2 = np.zeros((height,wide))
    temp_file3 = np.zeros((height, wide))
    temp_file4 = np.zeros((height,wide))
    gt_list = []
    temp_file1[mask1] = 1
    gt_list.append(temp_file1)
    temp_file2[mask2] = 1
    temp_file2[mask1] = 1
    gt_list.append(temp_file2)
    temp_file3[mask1] = 1
    temp_file3[mask2] = 1
    temp_file3[mask3] = 1
    gt_list.append(temp_file3)
    temp_file4[mask1] = 1
    temp_file4[mask2] = 1
    temp_file4[mask3] = 1
    temp_file4[mask4] = 1
    gt_list.append(temp_file4)
    gt_list = np.stack(gt_list, axis=0)
    return gt_list




# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        print("len_image",len(self.images))
        print("len_gt", len(self.gts))
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.ToTensor()])
        self.bin_transform = transforms.Compose(
            [transforms.ToTensor()])
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.binary_loader(self.depths[index])
        gt_trans = np.array(gt, dtype=np.uint8).copy()
        depth_trans = np.array(depth, dtype=np.uint8).copy()
        bins_mask = get_bins_masks(depth_trans)
        bins_depth = bin_trans(bins_mask, depth_trans)
        bins_depth = transforms.ToPILImage()(bins_depth)

        gt_stack = get_gt_stack(gt_trans)
        gt_stack = torch.from_numpy(gt_stack).float()
        gt_stack = transforms.ToPILImage()(gt_stack)
        # print("before", image.size)
        image,gt,depth,bins_depth,gt_stack=cv_random_flip(image,gt,depth,bins_depth,gt_stack)
        # print("after1", image.size)
        # image,gt,depth,bins_depth,gt_stack=randomCrop(image, gt,depth,bins_depth,gt_stack)
        # print("after2", image.size)
        image,gt,depth,bins_depth,gt_stack=randomRotation(image, gt,depth,bins_depth,gt_stack)
        # print("after", image.size)
        image=colorEnhance(image)
        # gt=randomGaussian(gt)
        gt=randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth=self.depths_transform(depth)
        bins_depth = self.depths_transform(bins_depth)
        gt_stack = self.gt_transform(gt_stack)
        # print(image.size, depth.size, gt.size)
        # #bins
        # depth_trans = cv2.normalize(np.asarray(depth), None, 0, 255, norm_type=cv2.NORM_MINMAX)
        # depth_trans = np.asarray(depth_trans).astype('uint8')
        # bins_mask = get_bins_masks(depth_trans)
        # depth_trans = torch.from_numpy(depth_trans).float()
        # depth_trans_2 = np.squeeze(depth_trans, axis=0)
        # depth_trans_2 = Image.fromarray(np.asarray(depth_trans_2).astype('uint8')).convert("L")
        # depth_trans_2 = torch.from_numpy(np.asarray(depth_trans_2)).float()
        # bins_mask = torch.from_numpy(bins_mask).float()
        # h, w = depth[0].size()
        # bins_depth = depth_trans_2.view(1, h, w).repeat(3, 1, 1)
        # print(bins_mask)
        # bins_depth = bins_depth * bins_mask
        # for i in range(3):
        #     bins_depth[i] = bins_depth[i] / bins_depth[i].max()
        # print("depth_bin_______________",bins_depth)
        # bins_mask = torch.from_numpy(bins_mask).float()
        # depth_255 = depth_255.astype(np.float64)/255.0
        # depth_255 = torch.from_numpy(depth_255).float()
        # print(depth_255.size())
        # h, w = depth_255.size()
        # bins_depth = depth_255.view(1, h, w).repeat(3, 1, 1)
        # bins_depth = bins_depth * bins_mask
        # for i in range(3):
        #     bins_depth[i] = bins_depth[i] / bins_depth[i].max()
        return image, gt, depth, bins_depth, gt_stack

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.depths)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, depth_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root,depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth=self.binary_loader(self.depths[self.index])
        gt_trans = np.array(gt, dtype=np.uint8).copy()
        depth_trans = np.array(depth, dtype=np.uint8).copy()
        gt_stack = get_gt_stack(gt_trans)
        gt_stack = torch.from_numpy(gt_stack).float()
        gt_stack = transforms.ToPILImage()(gt_stack)
        bins_mask = get_bins_masks(depth_trans)
        bins_depth = bin_trans(bins_mask, depth_trans)
        bins_depth = transforms.ToPILImage()(bins_depth)
        bins_depth = self.depths_transform(bins_depth).unsqueeze(0)
        depth=self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)

        # depth_trans = cv2.normalize(np.asarray(depth[0]), None, 0, 255, norm_type=cv2.NORM_MINMAX)
        # if (len(depth) > 1):
        #     for depth_1 in range(1, len(depth)):
        #         temp = cv2.normalize(np.asarray(depth[depth_1]), None, 0, 255, norm_type=cv2.NORM_MINMAX)
        #         depth_trans = torch.cat((depth_trans, temp), 2)
        # print(depth_trans)
        # depth_trans = np.array(depth_trans, dtype=np.int32)
        # print(depth_trans)
        # bins_mask = get_bins_masks(depth_trans)
        # bins_mask = torch.from_numpy(bins_mask).float()
        # print(depth[0].size())
        # h, w = depth[0].size()
        # bins_depth = depth.view(1, h, w).repeat(3, 1, 1)
        # bins_depth = bins_depth * bins_mask
        # for i in range(3):
        #     bins_depth[i] = bins_depth[i] / bins_depth[i].max()
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt,depth, name,np.array(image_for_post), bins_depth, gt_stack

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

