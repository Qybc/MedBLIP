from collections import defaultdict
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

import SimpleITK as sitk

class ImageTextContrastiveDataset(Dataset):

    def __init__(self, datalist=['ADNI-train']) -> None:
        super().__init__()
        # imgpath, report
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}.csv'
            print('load data from', filename)
            with open(filename) as f:
                lines = f.readlines()
                for line in lines:
                    imgpath,report = line.strip('\n').split('\t')
                    df_list.append((imgpath,report))

        self.df = df_list

    def pad_img(self, img, size=224):
        '''pad img to square.
        '''
        x, y, z = img.shape
        img = img.unsqueeze(0).unsqueeze(0) # BCHWD
        max_size = max(x, y, z)
        new_size = (int(size*x/max_size), int(size*y/max_size), int(size*z/max_size))
        img = F.interpolate(img,size=new_size,mode='trilinear',align_corners=True)

        x,y,z = new_size
        new_im = torch.zeros((1,1,size,size,size))
        x_min = int((size - x) / 2)
        x_max = x_min + x
        y_min = int((size - y) / 2)
        y_max = y_min + y
        z_min = int((size - z) / 2)
        z_max = z_min + z
        new_im[:,:,x_min:x_max,y_min:y_max,z_min:z_max] = img
        
        return new_im
    
    def norm_img(self, img):
        return (img - img.min())/(img.max() - img.min())

    def __getitem__(self, index):
        imgpath,report = self.df[index]
        img = torch.FloatTensor(sitk.GetArrayFromImage(sitk.ReadImage(imgpath)).astype(float))
        img = self.norm_img(img)
        img = self.pad_img(img)
        
        return img, report

    def __len__(self):
        return len(self.df)

class ImageTextContrastiveCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['images'].append(data[0])
            inputs['reports'].append(data[1])

        inputs['images'] = torch.cat(inputs['images'], 0)

        return inputs

class ZeroShotImageDataset(Dataset):
    def __init__(self,
        datalist=['ADNI-train']
        ) -> None:
        super().__init__()

        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}.csv'
            print('load data from', filename)
            with open(filename) as f:
                lines = f.readlines()
                for line in lines:
                    imgpath,report = line.strip('\n').split('\t')
                    df_list.append((imgpath,report))
        self.df = df_list

    def pad_img(self, img, size=224):
        '''pad img to square.
        '''

        x, y, z = img.shape
        img = img.unsqueeze(0).unsqueeze(0) # BCHWD
        max_size = max(x, y, z)
        new_size = (int(size*x/max_size), int(size*y/max_size), int(size*z/max_size))
        img = F.interpolate(img,size=new_size,mode='trilinear',align_corners=True)

        x,y,z = new_size
        new_im = torch.zeros((1,1,size,size,size))
        x_min = int((size - x) / 2)
        x_max = x_min + x
        y_min = int((size - y) / 2)
        y_max = y_min + y
        z_min = int((size - z) / 2)
        z_max = z_min + z
        new_im[:,:,x_min:x_max,y_min:y_max,z_min:z_max] = img
        
        return new_im
    
    def norm_img(self, img):
        return (img - img.min())/(img.max() - img.min())


    def __getitem__(self, index):
        imgpath,report = self.df[index]
        img = torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(imgpath)).astype(float))
        img = self.norm_img(img)
        img = self.pad_img(img)

        return img, report

    def __len__(self):
        return len(self.df)

class ZeroShotImageCollator:
    def __init__(self):
        return
    
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['images'].append(data[0])
            inputs['reports'].append(data[1])

        inputs['images'] = torch.cat(inputs['images'], 0)

        return inputs

