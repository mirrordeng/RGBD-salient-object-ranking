import torch
import torch.nn.functional as F
import sys
sys.path.append("./BBSNet_cpts/")
import numpy as np
import os, argparse
import cv2
from models.BBSNet_model import BBSNet
from data import test_dataset
import scipy.stats as sc

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='../BBS_dataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = BBSNet()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load("/nfs/volume-382-84/jasondeng_i/sor/bbsnet/BBS-Net13/BBSNet_cpts/BBSNet_epoch_best_mae.pth"))
model.cuda()
model.eval()
def get_norm_spr(spr_value):
    """
    Examples
    --------
    sum += get_norm_spr(t)
    """
    #       m - r_min
    # m -> ---------------- x (t_max - t_min) + t_min
    #       r_max - r_min
    #
    # m = measure value
    # r_min = min range of measurement
    # r_max = max range of measurement
    # t_min = min range of desired scale
    # t_max = max range of desired scale

    r_min = -1
    r_max = 1

    norm_spr = (spr_value - r_min) / (r_max - r_min)

    return norm_spr


def count_sor(gt, prediction):
    mask_rank1 = (gt == 255)
    mask_rank2 = (gt == 170)
    mask_rank3 = (gt == 85)
    sor_rank1 = np.mean(prediction[mask_rank1])
    sor_rank2 = np.mean(prediction[mask_rank2])
    sor_rank3 = np.mean(prediction[mask_rank3])
    prediction_order = [sor_rank1, sor_rank2, sor_rank3]
    gt_order = [3, 2, 1]
    spr = sc.spearmanr(gt_order, prediction_order)
    if np.isnan(sc.spearmanr(gt_order, prediction_order)[0]):
        print("WRONG", gt_order, prediction_order, prediction.max(),prediction.min(), gt.max(),gt.min())
        if np.isnan(prediction_order[0]):
            prediction_order = [sor_rank2, sor_rank3]
            gt_order = [2, 1]
            spr = sc.spearmanr(gt_order, prediction_order)
            spr = np.float(spr[0])
            spr = get_norm_spr(spr)
        elif np.isnan(prediction_order[1]):
            prediction_order = [sor_rank1, sor_rank3]
            gt_order = [2, 1]
            spr = sc.spearmanr(gt_order, prediction_order)
            spr = np.float(spr[0])
            spr = get_norm_spr(spr)
        elif np.isnan(prediction_order[2]):
            prediction_order = [sor_rank1, sor_rank2]
            gt_order = [2, 1]
            spr = sc.spearmanr(gt_order, prediction_order)
            spr = np.float(spr[0])
            spr = get_norm_spr(spr)
        if not(np.isnan(sor_rank1) and np.isnan(sor_rank2) and np.isnan(sor_rank3)):
            if np.isnan(sor_rank1) and np.isnan(sor_rank2):
                spr = 1
            elif np.isnan(sor_rank2) and np.isnan(sor_rank3): 
                spr =1
            elif np.isnan(sor_rank1) and np.isnan(sor_rank3): 
                spr =1
        else:
            spr  =  0
    else:        
        spr = np.float(spr[0])
        spr = get_norm_spr(spr)
    return spr

def count_sor_new(gt, prediction):
    mask_list = []
    gt_order = []
    sor_list = []
    for i in range(1, 5):
        mask_temp = (gt == i)
        if True in mask_temp:
            mask_list.append(mask_temp)
            gt_order.append(i)
    print( "gt_order", gt_order)
    for i in range(len(gt_order)):
        sor_temp = np.mean(prediction[mask_list[i]])
        sor_list.append(sor_temp)
    if len(gt_order) == 1:
        return 1
    if len(gt_order) == 0:
        return 100
    print("prediction", sor_list)
    spr = sc.spearmanr(gt_order, sor_list)
    # if np.isnan(sc.spearmanr(gt_order, prediction_order)[0]):
    #     print("WRONG", gt_order, prediction_order, prediction.max(),prediction.min(), gt.max(),gt.min())
    #     if np.isnan(prediction_order[0]):
    #         prediction_order = [sor_rank2, sor_rank3]
    #         gt_order = [2, 1]
    #         spr = sc.spearmanr(gt_order, prediction_order)
    #         spr = np.float(spr[0])
    #         spr = get_norm_spr(spr)
    #     elif np.isnan(prediction_order[1]):
    #         prediction_order = [sor_rank1, sor_rank3]
    #         gt_order = [2, 1]
    #         spr = sc.spearmanr(gt_order, prediction_order)
    #         spr = np.float(spr[0])
    #         spr = get_norm_spr(spr)
    #     elif np.isnan(prediction_order[2]):
    #         prediction_order = [sor_rank1, sor_rank2]
    #         gt_order = [2, 1]
    #         spr = sc.spearmanr(gt_order, prediction_order)
    #         spr = np.float(spr[0])
    #         spr = get_norm_spr(spr)
    #     if not(np.isnan(sor_rank1) and np.isnan(sor_rank2) and np.isnan(sor_rank3)):
    #         if np.isnan(sor_rank1) and np.isnan(sor_rank2):
    #             spr = 1
    #         elif np.isnan(sor_rank2) and np.isnan(sor_rank3): 
    #             spr =1
    #         elif np.isnan(sor_rank1) and np.isnan(sor_rank3): 
    #             spr =1
    #     else:
    #         spr  =  0
    # else:        
    spr = np.float(spr[0])
    spr = get_norm_spr(spr)
    return spr

#test
test_datasets = ['test_in_train']
for dataset in test_datasets:
    spr_count = 0
    mae_sum = 0
    save_path = './test_maps/BBSNet/ResNet50/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT_new/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    waste = 0
    for i in range(test_loader.size):
        image, gt, depth, name, img_for_post, bins_mask, gt_stack = test_loader.load_data()
        image = image.cuda()
        depth = depth.cuda()
        bins_mask=bins_mask.cuda()
        gt2 = np.asarray(gt, np.float32)
        _,res_,res_1,res_2,res = model(image, depth, bins_mask)
        res_ = res_.sigmoid().data.cpu().numpy().squeeze()
        res_ = (res_ - res_.min()) / (res_.max() - res_.min() + 1e-8)
        res_1 = res_1.sigmoid().data.cpu().numpy().squeeze()
        res_1 = (res_1 - res_1.min()) / (res_1.max() - res_1.min() + 1e-8)
        res_2 = res_2.sigmoid().data.cpu().numpy().squeeze()
        res_2= (res_2 - res_2.min()) / (res_2.max() - res_2.min() + 1e-8)
        _ = _.sigmoid().data.cpu().numpy().squeeze()
        _ = (_ - _.min()) / (_.max() - _.min() + 1e-8)
        res_final = res.sigmoid().data.cpu().numpy().squeeze()
        res_all = (res_final[0] - res_final[0].min()) / (res_final[0].max() - res_final[0].min() + 1e-8)
        for i in  range(1,4):
            res_all += (res_final[i] - res_final[i].min()) / (res_final[i].max() - res_final[i].min() + 1e-8) 
        gt = np.asarray(gt)
        temp = count_sor_new(gt, res_all)
        if temp != 100:
            spr_count += temp
        else:
            waste += 1
        print(temp)
        res_all = res_all/4
        gt2 /= (gt2.max() + 1e-8)
        mae_sum += np.sum(np.abs(res_all - gt2)) * 1.0 / (gt2.shape[0] * gt2.shape[1])
        print('save img to: ',save_path+name)
        gt_stack =np.asarray(gt_stack, np.float32)
        print(gt_stack.shape)
        gt1 = gt_stack[:,:,0]
        gt_2 = gt_stack[:,:,1]
        gt3 = gt_stack[:,:,2]
        gt4 = gt_stack[:,:,3]
        cv2.imwrite(save_path+name,res_all*255)


        cv2.imwrite(save_path+name[:-4]+"gt_.png", gt2*255)

    mae = mae_sum / test_loader.size
    spr = spr_count / (test_loader.size- waste)
    print(waste)
    print("MAEΪ", mae)
    print("SprΪ", spr)
    print('Test Done!')
