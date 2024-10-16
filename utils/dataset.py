from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import cv2


class MorphDataset(Dataset):
    def __init__(self, datapath, image_size, phase='train', transform=None):
        assert datapath is not None
        if "FF++" in datapath:
            # dataset = "FF++"
            assert phase in ['train','val','test']
            datapath = os.path.join(datapath, phase)
        # elif 'FRLL' in datapath:
        #     dataset = "FRLL"
        # elif 'FRGC' in datapath:
        #     dataset = "FRGC"
        # elif 'FERET' in datapath:
        #     dataset = "FERET"
        
        labels = []
        image_paths = []
        for method in os.listdir(datapath):
            for root, _, files in os.walk(os.path.join(datapath, method)):
                if not files:
                    continue
                for filename in files:
                    if filename.endswith(('.png', '.jpg')):
                        image_paths.append(os.path.join(root, filename))
                        if "real" in root or "raw" in root:
                            labels.append(0)
                        else:
                            labels.append(1)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        img=np.array(Image.open(self.image_paths[idx]))
        label=self.labels[idx]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img=img.transpose((2,0,1))
        img = img.astype('float32')/255
        if self.transform:
            img = self.transform(img)
        return img, label
    

class PartialMorphDataset(Dataset):
    def __init__(self, datapath, image_size, transform=None, method=None):
        assert datapath is not None
        assert method is not None
        if 'FRLL' in datapath:
            assert method in ['amsl', 'facemorpher', 'opencv', 'stylegan', 'webmorph']
            self.dataset_name = f"FRLL_{method}"
        elif 'FRGC' in datapath:
            assert method in ['facemorpher', 'opencv', 'stylegan']
            self.dataset_name = f"FRGC_{method}"
        elif 'FERET' in datapath:
            assert method in ['facemorpher', 'opencv', 'stylegan']
            self.dataset_name = f"FERET_{method}"


        labels = []
        image_paths = []
        for curr_method in os.listdir(datapath):
            for root, _, files in os.walk(os.path.join(datapath, curr_method)):
                if not files:
                    continue
                for filename in files:
                    if filename.endswith(('.png', '.jpg')) and method in root or "real" in root or "raw" in root:
                        image_paths.append(os.path.join(root, filename))
                        if "real" in root or "raw" in root:
                            labels.append(0)
                        else:
                            labels.append(1)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        img=np.array(Image.open(self.image_paths[idx]))
        label=self.labels[idx]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img=img.transpose((2,0,1))
        img = img.astype('float32')/255
        if self.transform:
            img = self.transform(img)
        return img, label
    
# class MorDIFF(Dataset):
#     def __init__(self, datapath_fake, datapath_real, image_size, transform=None):
#         assert datapath_fake is not None
#         assert datapath_real is not None
        
#         labels = []
#         image_paths = []
#         for curr_faces in os.listdir(datapath_fake):
#             for root, _, files in os.walk(os.path.join(datapath_fake, curr_faces)):
#                 if not files:
#                     continue
#                 for filename in files:
#                     if filename.endswith(('.png', '.jpg')):
#                         image_paths.append(os.path.join(root, filename))
#                         labels.append(1)

#         for filename in os.listdir(os.path.join(datapath_real, 'FRLL-Morphs_cropped', 'raw')):
#             if filename.endswith(('.png', '.jpg')):
#                 image_paths.append(os.path.join(datapath_real, 'FRLL-Morphs_cropped', 'raw', filename))
#                 labels.append(0)

#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform
#         self.img_size = image_size

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
        
#         img=np.array(Image.open(self.image_paths[idx]))
#         label=self.labels[idx]
#         img = cv2.resize(img, (self.img_size, self.img_size))
#         img=img.transpose((2,0,1))
#         img = img.astype('float32')/255
#         if self.transform:
#             img = self.transform(img)
#         return img, label