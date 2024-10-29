import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import torchio as tio
NUM_WORKERS = 8
PIN_MEMORY = True
DIM_ = 256
def resample_itk_image_LA(itk_image):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    #out_spacing = (1,1,1)   # For CAMUS Data 
    out_spacing = (1.25,1.25,1)   # For MnM2  Data 
    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]
    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image

def crop_center(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_Function(org_dim3,org_dim1,org_dim2,DIM_,img_):# org_dim3->numof channels
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center(temp)   
    return img_


def Normalization_Image(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

geometrical_transforms = tio.OneOf([
    tio.RandomFlip(axes=([1, 2])),
    tio.RandomElasticDeformation(num_control_points=(5, 5, 5), locked_borders=1, image_interpolation='nearest'),
    tio.RandomAffine(degrees=(-45, 45), center='image'),
])

intensity_transforms = tio.OneOf([
    tio.RandomBlur(),
    tio.RandomGamma(log_gamma=(-0.2, -0.2)),
    tio.RandomNoise(mean=0.1, std=0.1),
    tio.RandomGhosting(axes=([1, 2])),
])

transforms_2d = tio.Compose({
    geometrical_transforms: 0.3,  # Probability for geometric transforms
    intensity_transforms: 0.3,   # Probability for intensity transforms
    tio.Lambda(lambda x: x): 0.4 # Probability for no augmentation (original image)
})
   
def generate_label(gt):
        temp_ = np.zeros([4,DIM_,DIM_])
        temp_[0:1,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:][np.where(gt==0)]=1
        return temp_

def gen_meta_MnM2(vendor, scanners, disease, field, size):   ## This is for MnM2 MetaData
    temp = np.zeros(size)
    # Mapping vendors to numerical values
    vendor_mapping = {'Philips Medical Systems': 1, 'SIEMENS': 2, 'GE MEDICAL SYSTEMS': 3}
    temp[0] = vendor_mapping.get(vendor, 0)
    # Mapping scanners to numerical values
    scanners_mapping = {
        'Symphony': 4, 'SIGNA EXCITE': 3, 'Signa Explorer': 2,
        'SymphonyTim': 1, 'Avanto Fit':0 , 'Avanto': -1,
        'Achieva': -2, 'Signa HDxt': -3, 'TrioTim': -4
    }
    temp[1] = scanners_mapping.get(scanners, 0)
    # Mapping diseases to numerical values
    disease_mapping = {'NOR': 3, 'LV': 2, 'HCM': 1, 'ARR': 0, 'FALL': -1, 'CIA': -2}
    temp[2] = disease_mapping.get(disease, 0)
    # Mapping field to numerical values
    temp[3] = float(field)
    return temp
vendor_gt = {
    'GE MEDICAL SYSTEMS': 0,
    'SIEMENS': 1,
    'Philips Medical Systems': 2,
}
scanner_gt = {
        'Symphony': 0, 'SIGNA EXCITE': 1, 'Signa Explorer': 2,
        'SymphonyTim': 3, 'Avanto Fit': 4, 'Avanto': 5,
        'Achieva': 6, 'Signa HDxt': 7, 'TrioTim': 8
    }
disease_gt = {'NOR': 0, 'LV': 1, 'HCM': 2, 'ARR': 3, 'FALL': 4, 'CIA': 5}
field_gt = {'1.5': 0, '3.0': 1}


sex_mapping = {'M': 0, 'F': 1}
quality_mapping = {'Good': 0, 'Medium': 1, 'Poor': 2}
def gen_meta_CAMUS(sex, image_quality,es,nb_frame,age,ef,frame_rate, size):   ## This is for CAMUS MetaData
    temp = np.zeros(size,dtype=float)
    temp[0] = float(sex_mapping[sex])
    temp[1] = float(quality_mapping[image_quality])
    temp[2] = float(es)
    temp[3] = float(nb_frame)
    temp[4] = float(age)
    temp[5] = float(ef)
    temp[6] = float(frame_rate)
    return temp


class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases = df['DISEASE']
        self.fields = df['FIELD']      
        self.images_name1 = df['SUBJECT_CODE'] 
        
        self.transformations = transformations
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_Image(img)
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_Function(C,H,W,256,img)
        img = np.expand_dims(img, axis=3)
        
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_Function(C,H,W,256,gt)
        gt = np.expand_dims(gt, axis=3)
        
        d = {}
        d['Image'] = tio.Image(tensor = img, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = gt, type=tio.LABEL)
        sample = tio.Subject(d)
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img = transformed_tensor['Image'].data
            gt = transformed_tensor['Mask'].data
    
        gt = gt[...,0]
        img = img[...,0] 
        gt_super = np.zeros_like(gt)
        gt_super[np.where(gt!=0)]=1
        gt_super = np.concatenate((gt_super, (1-gt_super)), axis=0)

        gt_sub = generate_label(gt)
        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        scanner = self.scanners[ind]
        disease = self.diseases[ind]
        field = self.fields[ind]
        
        v_gt = vendor_gt.get(vendor, 0)
        s_gt = scanner_gt.get(scanner, 0)
        d_gt = disease_gt.get(disease, 0)
        f_gt = field_gt.get(str(field), 0)
        meta_16 = gen_meta_MnM2(vendor, scanner, disease, field, (4))

        return img,gt_sub,gt_super,meta_16,v_gt,s_gt,d_gt,f_gt
        
def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


class Dataset_val(Dataset): 
    def __init__(self, df, images_folder):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases = df['DISEASE']
        self.fields = df['FIELD']      
        self.images_name1 = df['SUBJECT_CODE'] 
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_Image(img)
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_Function(C,H,W,256,img)
        
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_Function(C,H,W,256,gt)
        gt_super = np.zeros_like(gt)
        gt_super[np.where(gt!=0)]=1
        gt_super = np.concatenate((gt_super, (1-gt_super)), axis=0)
        
        gt_sub = generate_label(gt)
        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        scanner = self.scanners[ind]
        disease = self.diseases[ind]
        field = self.fields[ind]
        
        v_gt = vendor_gt.get(vendor, 0)
        s_gt = scanner_gt.get(scanner, 0)
        d_gt = disease_gt.get(disease, 0)
        f_gt = field_gt.get(str(field), 0)
        meta_16 = gen_meta_MnM2(vendor, scanner, disease, field, (4))
                
        return img,gt_sub,gt_super,meta_16,v_gt,s_gt,d_gt,f_gt
        
def Data_Loader_val(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader
