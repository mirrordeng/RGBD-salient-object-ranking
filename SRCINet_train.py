import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.BBSNet_model import BBSNet
from data import get_loader,test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from torchsummary import summary
import scipy.stats as sc
from torchvision import models

#set the device for training
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

def model_info(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = model.to(device)
    summary(backbone,torch.randn((1,3,480,640)),torch.randn((1,1,480,640)),torch.randn((1,3,480,640)))

#build the model
model = BBSNet()
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ',opt.load)
    
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
#set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root=opt.depth_root
test_image_root=opt.test_rgb_root
test_gt_root=opt.test_gt_root
test_depth_root=opt.test_depth_root
save_path=opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(image_root, gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root,test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))

#set loss function
CE = torch.nn.BCEWithLogitsLoss()
step=0
writer = SummaryWriter(save_path+'summary')
best_mae=1
best_epoch=0

#train function
def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths, bins_mask, gt_stack) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            
            images = images.cuda()
            gts = gts.cuda()
            depths=depths.cuda()
            bins_mask=bins_mask.cuda()
            gt_stack=gt_stack.cuda()
            s1, s2_1, s2_2, s2_3, s3 = model(images,depths, bins_mask, gumbel=True)
            loss1 = CE(s1, gt_stack[:,3,:,:].unsqueeze(1))
            loss2_1 =CE(s2_1,gt_stack[:,0,:,:].unsqueeze(1))
            loss2_2 = CE(s2_2,gt_stack[:,1,:,:].unsqueeze(1))
            loss2_3 = CE(s2_3, gt_stack[:,2,:,:].unsqueeze(1))
            loss3 = CE(s3, gt_stack)
            # loss3 = SmoothL1(s3*50, gts*50)
            loss = loss1+ loss2_1+loss2_2+loss2_3  +loss3    #+loss3
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data
            s3_1 = s3[0]
            for i in  range(1,4):
                s3_1 += s3[i] 
            if i % 100 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2_1: {:0.4f} Loss2_2: {:0.4f} Loss2_3: {:0.4f}, loss2_4: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2_1.data, loss2_2.data, loss2_3.data, loss2_4.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2_1: {:0.4f} Loss2_2: {:0.4f} Loss2_3: {:0.4f}'.
                    format( epoch, opt.epoch, i, total_step, loss1.data, loss2_1.data, loss2_2.data, loss2_3.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res=s1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s1', torch.tensor(res), step,dataformats='HW')
                res_1=s2_1[0].clone()
                res_1 = res_1.sigmoid().data.cpu().numpy().squeeze()
                res_1 = (res_1 - res_1.min()) / (res_1.max() - res_1.min() + 1e-8)
                writer.add_image('s2_1', torch.tensor(res_1), step,dataformats='HW')
                res_2 = s2_2[0].clone()
                res_2 = res_2.sigmoid().data.cpu().numpy().squeeze()
                res_2 = (res_2 - res_2.min()) / (res_2.max() - res_2.min() + 1e-8)
                writer.add_image('s2_2', torch.tensor(res_2), step, dataformats='HW')
                res_3 = s2_3[0].clone()
                res_3 = res_3.sigmoid().data.cpu().numpy().squeeze()
                res_3 = (res_3 - res_3.min()) / (res_3.max() - res_3.min() + 1e-8)
                writer.add_image('s2_3', torch.tensor(res_3), step, dataformats='HW')
                res_final = s3_1[0].clone()
                res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
                res_final = (res_final - res_final.min()) / (res_final.max() - res_final.min() + 1e-8)
                writer.add_image('s3', torch.tensor(res_final), step, dataformats='HW')

        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path+'BBSNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'BBSNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
def get_norm_spr(spr_value):
    """该函数用于计算归一化到[0,1]区间的斯皮尔曼系数

    :param:spr_value,原斯皮尔曼系数
    :return:norm_spr，归一化后的斯皮尔曼系数

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

def count_sor(gt,  prediction):
    mask_rank1 = (gt == 255)
    mask_rank2 = (gt == 170)
    mask_rank3 = (gt == 85)
    sor_rank1 = np.mean(prediction[mask_rank1])
    sor_rank2  = np.mean(prediction[mask_rank2])
    sor_rank3 = np.mean(prediction[mask_rank3])
    prediction_order = [sor_rank1, sor_rank2, sor_rank3]
    gt_order = [3, 2, 1]
    spr = sc.spearmanr(gt_order, prediction_order)
    spr = np.float(spr[0])
    spr = get_norm_spr(spr)
    return spr

#test function
def test(test_loader,model,epoch,save_path):
    global best_mae,best_epoch_mae, best_epoch_sor,best_sor
    model.eval()
    with torch.no_grad():
        mae_sum=0
        spr_sum = 0
        for i in range(test_loader.size):
            image, gt,depth, name,img_for_post, bins_mask, gt_stack = test_loader.load_data()
            gt2 = np.asarray(gt)
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            gt_stack = np.asarray(gt_stack, np.float32)
            image = image.cuda()
            depth = depth.cuda()
            bins_mask = bins_mask.cuda()
            _,res,res_1,res_2,res_final= model(image,depth, bins_mask)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_all = res_final[0]
            for i in  range(1,4):
                res_temp = (res_final[i] - res_final[i].min()) / (res_final[i].max() - res_final[i].min() + 1e-8)
                res_all += res_temp
            res_all = (res_all - res_all.min()) / (res_all.max() - res_all.min() + 1e-8)
            spr_sum += count_sor(gt, res_all*4)
            mae_sum+=np.sum(np.abs(res_all-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae=mae_sum/test_loader.size
        spr = spr_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        writer.add_scalar('SOR', torch.tensor(spr), global_step=epoch)
        print('Epoch: {} MAE: {}  SOR: {}####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,spr,best_mae,best_epoch))
        if epoch==1:
            best_mae=mae
            best_sor = spr
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch_mae=epoch
                torch.save(model.state_dict(), save_path+'BBSNet_epoch_best_mae.pth')
                print('best epoch:{}'.format(epoch))
            if spr>best_sor:
                best_sor=spr
                best_epoch_sor=epoch
                torch.save(model.state_dict(), save_path+'BBSNet_epoch_best_sor.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':
    #model_info(model)
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr=adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch,save_path)
        test(test_loader,model,epoch,save_path)
