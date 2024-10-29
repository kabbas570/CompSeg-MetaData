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
from Proposed_Approach import Model
from Data_Loader import Data_Loader_io_transforms,Data_Loader_val
def check_Dice_Score(loader, model1, device=DEVICE):
    Dice_score_LV = 0
    Dice_score_MYO = 0
    Dice_score_LA = 0
    correct_sex = 0
    correct_Quality = 0
    num_samples = 0
    loop = tqdm(loader)
    model1.eval()
    for batch_idx, (img,gt_sub,gt_super,temp) in enumerate(loop):
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt_sub = gt_sub.to(device=DEVICE,dtype=torch.float)  
        gt_super = gt_super.to(device=DEVICE,dtype=torch.float)  
        temp = temp.to(device=DEVICE,dtype=torch.float)
        labels = temp.type(torch.LongTensor) 
        labels = labels.to(DEVICE)
        with torch.no_grad(): 
            _,logits4, logits_sex, logits_img_Q, logits_es, logits_nb_frame, logits_nb_age, logits_ef, logits_frame_rate = model1(img,temp)
            ## classification ###
            _, predictions_sex = logits_sex.max(1)
            correct_sex += (predictions_sex == labels[:,0]).sum()
            num_samples += predictions_sex.size(0)
            _, predictions_Q = logits_img_Q.max(1)
            correct_Quality += (predictions_Q == labels[:,1]).sum()
            ## segemntaiton ##
            pred = torch.argmax(logits4, dim=1)
            out_LV = torch.zeros_like(pred)
            out_LV[torch.where(pred==0)] = 1 
            out_MYO = torch.zeros_like(pred)
            out_MYO[torch.where(pred==1)] = 1
            out_RV = torch.zeros_like(pred)
            out_RV[torch.where(pred==2)] = 1        
            single_LV = (2 * (out_LV * gt_sub[:,0,:]).sum()) / (
               (out_LV + gt_sub[:,0,:]).sum() + 1e-8)
            Dice_score_LV +=single_LV
            single_MYO = (2 * (out_MYO * gt_sub[:,1,:]).sum()) / (
   (out_MYO + gt_sub[:,1,:]).sum() + 1e-8)
            Dice_score_MYO += single_MYO
            single_LA = (2 * (out_RV * gt_sub[:,2,:]).sum()) / (
       (out_RV + gt_sub[:,2,:]).sum() + 1e-8)
            Dice_score_LA += single_LA
    ## classification ###
    print(f"Sex Accuracy  : {correct_sex/num_samples }")
    print(f"Quality Accuracy  : {correct_Quality/num_samples }")
    ## segemntaiton ##
    print(f"Dice_score_LV : {Dice_score_LV/len(loader)}")
    print(f"Dice_score_MYO  : {Dice_score_MYO/len(loader)}")
    print(f"Dice_score_LA  : {Dice_score_LA/len(loader)}")
    Overall_Dicescore = ( Dice_score_LV + Dice_score_MYO + Dice_score_LA)/3
    return Overall_Dicescore/len(loader)

def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler1): 
    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch
    train_losses1_class  = [] # loss of each batch
    valid_losses1_class = []  # loss of each batch
    train_losses1_Both  = [] # loss of each batch
    valid_losses1_Both = []  # loss of each batch
    loop = tqdm(loader_train1)
    model1.train()
    for batch_idx,(img,gt_sub,gt_super,temp)  in enumerate(loop):
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt_sub = gt_sub.to(device=DEVICE,dtype=torch.float)  
        gt_super = gt_super.to(device=DEVICE,dtype=torch.float)  
        temp = temp.to(device=DEVICE,dtype=torch.float)  
        labels = temp.type(torch.LongTensor) # <---- Here (casting)
        labels = labels.to(DEVICE)
        with torch.cuda.amp.autocast():
            logits2,logits4, logits_sex, logits_img_Q, logits_es, logits_nb_frame, logits_nb_age, logits_ef, logits_frame_rate  = model1(img,temp)    ## logits2, logits4, logits_v, logits_s, logits_d, logits_f 
            ## segmentation losses ##
            loss1 = loss_fn_seg(logits4,gt_sub)
            loss2 = loss_fn_seg(logits2,gt_super)
            loss_seg = 0.5*loss1+0.5*loss2
            ## classification  losses ##
            cl_loss_sex = loss_fn_cls(logits_sex,labels[:,0])
            cl_loss_Q = loss_fn_cls(logits_img_Q,labels[:,1])
            l1 = loss_fn_rgs(logits_es,labels[:,2:3])
            l2 = loss_fn_rgs(logits_nb_frame,labels[:,3:4])
            l3 = loss_fn_rgs(logits_nb_age,labels[:,4:5])
            l4 = loss_fn_rgs(logits_ef,labels[:,5:6])
            l5 = loss_fn_rgs(logits_frame_rate,labels[:,6:7])
            reg_loss = (l1+l2+l3+l4+l5)/5
            cl_loss = (cl_loss_sex+cl_loss_Q)/2
            loss = 0.7*loss_seg + 0.3*(cl_loss + reg_loss)
                        # backward
        optimizer1.zero_grad()        
        scaler1.scale(loss).backward()        
        scaler1.step(optimizer1)        
        scaler1.update()
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        train_losses1_seg.append(float(loss_seg))
        train_losses1_class.append(float(cl_loss))
        train_losses1_Both.append(float(loss))
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    for batch_idx,(img,gt_sub,gt_super,temp) in enumerate(loop_v):
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt_sub = gt_sub.to(device=DEVICE,dtype=torch.float)  
        gt_super = gt_super.to(device=DEVICE,dtype=torch.float) 
        temp = temp.to(device=DEVICE,dtype=torch.float)
        labels = temp.type(torch.LongTensor) # <---- Here (casting)
        labels = labels.to(DEVICE)
        with torch.no_grad(): 
            logits2,logits4, logits_sex, logits_img_Q, logits_es, logits_nb_frame, logits_nb_age, logits_ef, logits_frame_rate  = model1(img,temp) 
            ## segmentation losses ##
            loss1 = loss_fn_seg(logits4,gt_sub)
            loss2 = loss_fn_seg(logits2,gt_super)
            loss_seg = 0.5*loss1+0.5*loss2            
            ## classification  losses ##
            cl_loss_sex = loss_fn_cls(logits_sex,labels[:,0])
            cl_loss_Q = loss_fn_cls(logits_img_Q,labels[:,1])
            l1 = loss_fn_rgs(logits_es,labels[:,2:3])
            l2 = loss_fn_rgs(logits_nb_frame,labels[:,3:4])
            l3 = loss_fn_rgs(logits_nb_age,labels[:,4:5])
            l4 = loss_fn_rgs(logits_ef,labels[:,5:6])
            l5 = loss_fn_rgs(logits_frame_rate,labels[:,6:7])
            reg_loss = (l1+l2+l3+l4+l5)/5
            cl_loss = (cl_loss_sex+cl_loss_Q)/2
            loss = 0.7*loss_seg + 0.3*(cl_loss + reg_loss)
        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses1_seg.append(float(loss_seg))
        valid_losses1_class.append(float(cl_loss))
        valid_losses1_Both.append(float(loss))
    train_loss_per_epoch1_seg = np.average(train_losses1_seg)
    valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
    avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
    avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)
    train_loss_per_epoch1_class = np.average(train_losses1_class)
    valid_loss_per_epoch1_class  = np.average(valid_losses1_class)
    avg_train_losses1_class.append(train_loss_per_epoch1_class)
    avg_valid_losses1_class.append(valid_loss_per_epoch1_class)
    train_loss_per_epoch1_Both = np.average(train_losses1_Both)
    valid_loss_per_epoch1_Both  = np.average(valid_losses1_Both)
    avg_train_losses1_Both.append(train_loss_per_epoch1_Both)
    avg_valid_losses1_Both.append(valid_loss_per_epoch1_Both)
    return train_loss_per_epoch1_seg,valid_loss_per_epoch1_seg,train_loss_per_epoch1_class,valid_loss_per_epoch1_class,train_loss_per_epoch1_Both,valid_loss_per_epoch1_Both

for fold in range(1,6):
  fold = str(fold)  ## training fold number 
  train_imgs = "/pathF"+fold+"/train/imgs/"  
  val_imgs  = "/path/F"+fold+"/val/imgs/"
  ### Data is arranged as follows;
  # path/F1/train/imgs 
  #               /gts
  #       MetaData.csv ## MnM2 Data
  #         /cfg_files   ## CAMUS Data 
  # path/F2/train/imgs 
  #               /gts
  #       MetaData.csv ## MnM2 Data
  #         /cfg_files   ## CAMUS Data 
  # path/F3/train/imgs 
  #               /gts
  #       MetaData.csv ## MnM2 Data
  #         /cfg_files   ## CAMUS Data 
  # path/F4/train/imgs 
  #               /gts
  #       MetaData.csv ## MnM2 Data
  #         /cfg_files   ## CAMUS Data 
  # path/F5/train/imgs 
  #               /gts
  #       MetaData.csv ## MnM2 Data
  #         /cfg_files   ## CAMUS Data 
  
  Batch_Size = 16
  Max_Epochs = 500
  train_loader = Data_Loader_io_transforms(train_imgs,batch_size = Batch_Size)
  val_loader = Data_Loader_val(val_imgs,batch_size = 1)
  print(len(train_loader)) 
  print(len(val_loader))   
  avg_train_losses1_seg = []   # losses of all training epochs
  avg_valid_losses1_seg = []  #losses of all training epochs
  avg_valid_DS_ValSet_seg = []  # all training epochs
  avg_valid_DS_TrainSet_seg = []  # all training epochs
  avg_train_losses1_class = []   # losses of all training epochs
  avg_valid_losses1_class = []  #losses of all training epochs
  avg_valid_DS_ValSet_class = []  # all training epochs
  avg_valid_DS_TrainSet_class = []  # all training epochs
  avg_train_losses1_Both = []  #losses of all training epochs
  avg_valid_losses1_Both = []  #losses of all training epochs
  path_to_save_Learning_Curve = '/path/'+'/F'+fold+'Base_4_1'
  path_to_save_check_points = '/path/'+'/F'+fold+'Base_4_1'
  def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
      print("=> Saving checkpoint")
      torch.save(state, filename)
  model_1 = Model()
  epoch_len = len(str(Max_Epochs))
  def main():
      max_dice_val = 0.0
      model1 = model_1.to(device=DEVICE,dtype=torch.float)
      scaler1 = torch.cuda.amp.GradScaler()
      optimizer1 = optim.Adam(model1.parameters(),betas=(0.9, 0.99),lr=0.0001)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=[100,200,300,400,500], gamma=0.5)
      for epoch in range(Max_Epochs):
          train_loss_seg,valid_loss_seg ,train_loss_class,valid_loss_class ,train_loss_Both,valid_loss_Both = train_fn(train_loader,val_loader, model1, optimizer1,scaler1)
          scheduler.step()
          print_msg1 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
           f'train_loss_seg: {train_loss_seg:.5f} ' +
           f'valid_loss_seg: {valid_loss_seg:.5f}')
          print_msg2 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
           f'train_loss_class: {train_loss_class:.5f} ' +
           f'valid_loss_class: {valid_loss_class:.5f}')
          print_msg3 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
           f'train_loss_Both: {train_loss_Both:.5f} ' +
           f'valid_loss_Both: {valid_loss_Both:.5f}')
          print(print_msg1)
          print(print_msg2)
          print(print_msg3)
          Dice_val = check_Dice_Score(val_loader, model1, device=DEVICE)
          avg_valid_DS_ValSet_seg.append(Dice_val.detach().cpu().numpy())
          #avg_valid_DS_ValSet_class.append(Class_Acc.detach().cpu().numpy())
                      # save model 
          if Dice_val > max_dice_val:
              max_dice_val = Dice_val
                # Save the checkpoint
              checkpoint = {
                      "state_dict": model1.state_dict(),
                      "optimizer": optimizer1.state_dict(),
                      }
              save_checkpoint(checkpoint)
              
  if __name__ == "__main__":
      main()
  fig = plt.figure(figsize=(10,8))
  plt.plot(range(1,len(avg_train_losses1_seg)+1),avg_train_losses1_seg, label='Training Segmentation Loss')
  plt.plot(range(1,len(avg_valid_losses1_seg)+1),avg_valid_losses1_seg,label='Validation Segmentation Loss')
  plt.plot(range(1,len(avg_train_losses1_class)+1),avg_train_losses1_class, label='Training Classsification Loss')
  plt.plot(range(1,len(avg_valid_losses1_class)+1),avg_valid_losses1_class,label='Validation Classsification Loss')
  plt.plot(range(1,len(avg_train_losses1_Both)+1),avg_train_losses1_Both, label='Training Both')
  plt.plot(range(1,len(avg_valid_losses1_Both)+1),avg_valid_losses1_Both,label='Validation Both')
  plt.plot(range(1,len(avg_valid_DS_ValSet_seg)+1),avg_valid_DS_ValSet_seg,label='Validation DS')
  plt.plot(range(1,len(avg_valid_DS_ValSet_class)+1),avg_valid_DS_ValSet_class,label='Validation Acc')
    # find position of lowest validation loss
  minposs = avg_valid_losses1_seg.index(min(avg_valid_losses1_seg))+1 
  plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')
  font1 = {'size':20}
  plt.title("Learning Curve Graph",fontdict = font1)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.ylim(-1, 1) # consistent scale
  plt.xlim(0, len(avg_train_losses1_seg)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')
