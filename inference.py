import torch
import torch.nn as nn
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from Dice_Loss import DiceLoss
loss_fn_seg = DiceLoss()
loss_fn_cls = nn.CrossEntropyLoss()
loss_fn_rgs = nn.L1Loss()
from Data_Loader import Data_Loader_val
import torch
import torch.nn as nn
from medpy import metric
import kornia
def make_edges(image,three):
    three = np.stack((three,)*3, axis=2)
    three =torch.tensor(three)
    three = np.transpose(three, (2,0,1))  ## to bring channel first 
    three= torch.unsqueeze(three,axis = 0)
    magnitude, edges=kornia.filters.canny(three, low_threshold=0.1, high_threshold=0.2, kernel_size=(7, 7), sigma=(1, 1), hysteresis=True, eps=1e-06)
    image[np.where(edges[0,0,:,:]!=0)] = 1
    return image
def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
def blend(image,LV,MYO,RV,three): 
    image = normalize(image)
    image = np.stack((image,)*3, axis=2)
    image[np.where(LV==1)] = [0.9,0.9,0]
    image[np.where(MYO==1)] = [0.9,0,0]
    image[np.where(RV==1)] = [0,0,0.9]
    image = make_edges(image,three)
    return image
def calculate_metric_percase(pred, gt):
      gt = gt[0,:]
      pred = pred[0,:]
      dice = metric.binary.dc(pred, gt)
      hd = metric.binary.hd95(pred, gt)
      return dice, hd
def Average(lst): 
    return sum(lst) / len(lst)
Dice_LV_LA  = []
HD_LV_LA = []    
Dice_MYO_LA  = []
HD_MYO_LA = []   
Dice_RV_LA  = []
HD_RV_LA = []
for fold in range(1,6):
    fold = str(fold)  ## training fold number 
    path_to_checkpoints = '/weight_path/F'+fold+'Base_4_1.pth.tar'
    viz_gt_path = '/gt_viz_path/F'+fold+'/viz_gt/'
    viz_pred_path =  '/pred_viz_path/F'+fold+'/'
    val_imgs  = "/path/F"+fold+"/val/imgs/"
    Batch_Size = 1
    val_loader = Data_Loader_val(val_imgs,batch_size = Batch_Size)
    print(len(val_loader))   
    def check_Dice_Score(loader, model1, device=DEVICE):
        Dice_LV  = 0
        HD_LV = 0
        Dice_MYO  = 0
        HD_MYO = 0
        Dice_RV  = 0
        HD_RV = 0
        loop = tqdm(loader)
        model1.eval()
        for batch_idx,(img,g_sub,gt_super,temp,name) in enumerate(loop):
            img = img.to(device=DEVICE,dtype=torch.float)  
            g_sub = g_sub.to(device=DEVICE,dtype=torch.float)  
            gt_super = gt_super.to(device=DEVICE,dtype=torch.float)
            temp = temp.to(device=DEVICE,dtype=torch.float)
            labels = temp.type(torch.LongTensor) # <---- Here (casting)
            labels = labels.to(DEVICE)
            with torch.no_grad(): 
                _,pre_2d, _, _, _, _, _, _, _ = model1(img,temp)  
                pred = torch.argmax(pre_2d, dim=1)
                out_LV = torch.zeros_like(pred)
                out_LV[torch.where(pred==0)] = 1
                out_MYO = torch.zeros_like(pred)
                out_MYO[torch.where(pred==1)] = 1
                out_RV = torch.zeros_like(pred)
                out_RV[torch.where(pred==2)] = 1
                single_lv_dice,single_hd_lv = calculate_metric_percase(out_LV.detach().cpu().numpy(),g_sub[:,0,:].detach().cpu().numpy())
                single_myo_dice,single_hd_myo = calculate_metric_percase(out_MYO.detach().cpu().numpy(),g_sub[:,1,:].detach().cpu().numpy())
                single_rv_dice,single_hd_rv = calculate_metric_percase(out_RV.detach().cpu().numpy(),g_sub[:,2,:].detach().cpu().numpy())
                Dice_LV+=single_lv_dice
                HD_LV+=single_hd_lv
                Dice_MYO+=single_myo_dice
                HD_MYO+=single_hd_myo
                Dice_RV+=single_rv_dice
                HD_RV+=single_hd_rv
                img = img.detach().cpu().numpy()
                g_sub = g_sub.detach().cpu().numpy()
                out_LV = out_LV.detach().cpu().numpy()
                out_MYO = out_MYO.detach().cpu().numpy()
                out_RV = out_RV.detach().cpu().numpy()
                gt_blend = blend(img[0,0,:],g_sub[0,0,:],g_sub[0,1,:],g_sub[0,2,:],1-g_sub[0,3,:])
                plt.imsave(viz_gt_path +  name[0]  + '.png', gt_blend)
                pred_blend = blend(img[0,0,:],out_LV[0,:],out_MYO[0,:],out_RV[0,:],1-g_sub[0,3,:])
                plt.imsave(viz_pred_path +  name[0]  + '.png', pred_blend)
        print(' :: Dice Scores ::')
        print(f"Dice_LV  : {Dice_LV/len(loader)}")
        print(f"Dice_MYO  : {Dice_MYO/len(loader)}")
        print(f"Dice_RV  : {Dice_RV/len(loader)}")
        print(' :: HD Scores :: ')
        print(f"HD_LV  : {HD_LV/len(loader)}")
        print(f"HD_MYO  : {HD_MYO/len(loader)}")
        print(f"HD_RV  : {HD_RV/len(loader)}")
        print("                   ")
        print("Above is for fold -->", fold)
        return Dice_LV/len(loader), Dice_MYO/len(loader),Dice_RV/len(loader), HD_LV/len(loader),HD_MYO/len(loader),HD_RV/len(loader)
    model_1 = Model()
    def eval_():
        model = model_1.to(device=DEVICE,dtype=torch.float)
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0)
        checkpoint = torch.load(path_to_checkpoints,map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        Dice_LV, Dice_MYO,Dice_RV,HD_LV,HD_MYO,HD_RV= check_Dice_Score(val_loader, model, device=DEVICE)
        Dice_LV_LA.append(Dice_LV)
        Dice_MYO_LA.append(Dice_MYO)
        Dice_RV_LA.append(Dice_RV)
        HD_LV_LA.append(HD_LV)
        HD_MYO_LA.append(HD_MYO)
        HD_RV_LA.append(HD_RV)
    if __name__ == "__main__":
        eval_()
print("Average Five Fold Dice_LV_LA  --> ", Average(Dice_LV_LA))
print("Average Five Fold Dice_MYO_LA  --> ", Average(Dice_MYO_LA))
print("Average Five Fold Dice_RV_LA  --> ", Average(Dice_RV_LA))
print("Average Five Fold HD_LV_LA  --> ", Average(HD_LV_LA))
print("Average Five Fold HD_MYO_LA  --> ", Average(HD_MYO_LA))
print("Average Five Fold HD_RV_LA  --> ", Average(HD_RV_LA))
