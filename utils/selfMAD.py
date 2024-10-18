import torch
from torchvision import utils
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import random
import cv2
import albumentations as alb
import argparse

import warnings
warnings.filterwarnings('ignore')

class selfMAD_Dataset(Dataset):
	def __init__(self,phase='train',image_size=256,datapath=None):

		assert datapath is not None
		image_root = datapath
		landmark_root = image_root + "_landmarks"
		label_root = image_root + "_labels"
		if "FF++" in datapath:
			datapath = os.path.join(datapath, phase, "real")
			assert phase in ['train','val','test']	
		elif any(method in datapath for method in ['FRLL', 'FRGC', 'FERET']):
			datapath = os.path.join(datapath, "raw")
		elif 'SMDD' in datapath:
			datapath = os.path.join(datapath, "os25k_bf_t")

		image_list, landmark_list, label_list = [], [], []
		for root, _, files in os.walk(datapath):
			for filename in files:
				if filename.endswith(('.png', '.jpg')):
					img_path = os.path.join(root, filename)
					landmark_path = img_path.replace(image_root, landmark_root, -1).replace('.png', '.npy').replace('.jpg', '.npy')
					label_path = img_path.replace(image_root, label_root, -1).replace('.png', '.npy').replace('.jpg', '.npy')
					if not os.path.isfile(landmark_path) or not os.path.isfile(label_path):
						continue
					image_list.append(img_path)
					landmark_list.append(landmark_path)
					label_list.append(label_path)
	
		self.path_lm=landmark_list
		self.face_labels = label_list
		self.image_list=image_list
		self.image_size=(image_size,image_size)
		self.phase=phase
		self.transforms=self.get_transforms()
		self.source_transforms = self.get_source_transforms()
		# print(len(self.image_list), len(self.path_lm), len(self.face_labels))

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				filename=self.image_list[idx]
				img=np.array(Image.open(filename))
				landmark = np.load(self.path_lm[idx])
				landmark = landmark[0] if len(landmark) == 1 else landmark # FF => (81, 2); SMDD (1, 81, 2)
				landmark=self.reorder_landmark(landmark)

				bbox = np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()]).reshape(2,2)
				face_label = np.load(self.face_labels[idx])
			
				# random HFLIP
				if self.phase=='train':
					if np.random.rand()<0.5:
						img,_,landmark,bbox,face_label=self.hflip(img,None,landmark,bbox,face_label)
				
				img,landmark,bbox,face_label,_=self.crop_face(img,landmark,bbox,face_label,margin=True,crop_by_bbox=False,abs_coord=False,only_img=False)

				# BLENDING and MORPHING transforms
				if np.random.rand()<0.5:
					img_r,img_f, _ = self.self_blending(img.copy(),landmark.copy())
				else:
					img_r, img_f, _, _ = self.self_morphing(img.copy(), face_label.copy(), landmark.copy())

				if self.phase=='train':
					# Training augmentations
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']
					
					# Frequency transform
					if np.random.rand() < 0.1:
						freq_transform = self.create_frequency_noise_transform(weight=np.random.uniform(0.025, 0.1))
						img_f = freq_transform(image=img_f)['image']
				
				img_f,_,__,___, ____,y0_new,y1_new,x0_new,x1_new=self.crop_face(img_f,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)			
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				
				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255

				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))
				flag=False

			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img_f,img_r

	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)

	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		mask=np.zeros_like(img[:,:,0])
		cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=self.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None,face_label=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None

		if face_label is not None:
			face_label=face_label[:,::-1]

		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new,face_label
	
	def crop_face(self,img,landmark=None,bbox=None,face_label=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
		H,W=len(img),len(img[0])
		if crop_by_bbox:
			x0,y0=bbox[0]
			x1,y1=bbox[1]
			w=x1-x0
			h=y1-y0
			w0_margin=w/4
			w1_margin=w/4
			h0_margin=h/4
			h1_margin=h/4
		else:
			x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
			x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
			w=x1-x0
			h=y1-y0
			w0_margin=w/8
			w1_margin=w/8
			h0_margin=h/2
			h1_margin=h/5
		if margin:
			w0_margin*=4
			w1_margin*=4
			h0_margin*=2
			h1_margin*=2
		elif phase=='train':
			w0_margin*=(np.random.rand()*0.6+0.2)
			w1_margin*=(np.random.rand()*0.6+0.2)
			h0_margin*=(np.random.rand()*0.6+0.2)
			h1_margin*=(np.random.rand()*0.6+0.2)
		else:
			w0_margin*=0.5
			w1_margin*=0.5
			h0_margin*=0.5
			h1_margin*=0.5
				
		y0_new=max(0,int(y0-h0_margin))
		y1_new=min(H,int(y1+h1_margin)+1)
		x0_new=max(0,int(x0-w0_margin))
		x1_new=min(W,int(x1+w1_margin)+1)
		
		img_cropped=img[y0_new:y1_new,x0_new:x1_new]
		
		if face_label is not None:
			face_label = np.stack([face_label, face_label, face_label], axis=-1)
			face_label_cropped=face_label[y0_new:y1_new,x0_new:x1_new]
			face_label_cropped = face_label_cropped[:,:,0]
		else:
			face_label_cropped = None
		if landmark is not None:
			landmark_cropped=np.zeros_like(landmark)
			for i,(p,q) in enumerate(landmark):
				landmark_cropped[i]=[p-x0_new,q-y0_new]
		else:
			landmark_cropped=None
		if bbox is not None:
			bbox_cropped=np.zeros_like(bbox)
			for i,(p,q) in enumerate(bbox):
				bbox_cropped[i]=[p-x0_new,q-y0_new]
		else:
			bbox_cropped=None
		
		if only_img:
			return img_cropped
		if abs_coord:
			return img_cropped,landmark_cropped,bbox_cropped,face_label_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
		else:
			return img_cropped,landmark_cropped,bbox_cropped,face_label_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)
	
	def create_frequency_noise_transform(self, weight):
		return alb.Compose([
			FrequencyPatterns(p=1.0, mode=4, required_pattern=5, weight=weight, max_multi=3)
		])
	
	def zoom_at_point(self, img, x, y, zoom_factor):
		height, width, _ = img.shape
		new_width = int(width / zoom_factor)
		new_height = int(height / zoom_factor)
		left = max(0, x - new_width // 2)
		top = max(0, y - new_height // 2)
		right = min(width, x + new_width // 2)
		bottom = min(height, y + new_height // 2)

		if left == 0:
			right = new_width
		if right == width:
			left = width - new_width
		if top == 0:
			bottom = new_height
		if bottom == height:
			top = height - new_height

		cropped_image_array = img[top:bottom, left:right]
		zoomed_image_array = cv2.resize(cropped_image_array, (width, height), interpolation=cv2.INTER_LANCZOS4)
		return zoomed_image_array

	def self_morph(self, img, face_labels, landmarks, default_config={'zoom_factor': 1.05, 'zoom_center': 'landmark_30', 'mask_type': [0, 18, 17], 'blend_factor': 0.5}, config={}):
		config = {**default_config, **config}

		# MASK
		mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
		for label in config['mask_type']:
			if label == "full":
				break
			mask = mask & (face_labels != label)
		
		# CENTER OF ZOOM
		if 'landmark' in config['zoom_center']:
			landmark_idx = int(config['zoom_center'].split('_')[-1])
			center = landmarks[landmark_idx, :]
			assert center is not None, 'no landmark center provided'
		elif config['zoom_center'] == 'mask_mean':
			center = np.mean(np.argwhere(mask == 1), axis=0)[::-1].astype(int)
			
		mask = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
		source, target = img.copy(), img.copy()
		
		zoom_factor = config['zoom_factor']
		blend_factor = config['blend_factor']
		x, y = center

		zoomed_source = self.zoom_at_point(source, x, y,zoom_factor)
		zoomed_mask = self.zoom_at_point(mask, x, y, zoom_factor)

		img_f = target / 255
		zoomed_image_f = zoomed_source / 255

		morphed_img = np.where(zoomed_mask == 0, img_f, (1 - blend_factor) * img_f + blend_factor * zoomed_image_f)
		morphed_img = (morphed_img * 255).astype(np.uint8)
		bonafide_img = (img_f * 255).astype(np.uint8)

		return bonafide_img, morphed_img, center

	def self_morphing(self, img, face_labels, landmarks):
		config = {
			'zoom_factor': np.random.uniform(1.0, 1.1),
			'zoom_center': random.choice(['mask_mean', 'landmark_30']),
			'mask_type': random.choice([[0, 18, 17], [0, 13, 17, 18], [0, 13, 17, 18, 8, 9], [0, 17, 18, 8, 9], 
										[0, 17], [0, 13, 17], [0, 13, 17, 8, 9], [0, 17, 8, 9], 
										[18, 17], [13, 17, 18], [13, 17, 18, 8, 9], [17, 18, 8, 9],
										[17], [13, 17], [13, 17, 8, 9], [17, 8, 9],
										[0, 18], [0, 13, 18], [0, 13, 18, 8, 9], [0, 18, 8, 9], 
										[0], [0, 13], [0, 13, 8, 9], [0, 8, 9],
										[18], [13, 18], [13, 18, 8, 9], [18, 8, 9], 
										[13], [13, 8, 9], [8, 9],
										"full"
										]), 
			'blend_factor': random.choice([0.5, 0.5, 0.5, 0.375, 0.25, 0.125])
		}
		img_bonafide, morphed_img, center = self.self_morph(img, face_labels, landmarks, config)
		return img_bonafide, morphed_img, center, config

	def dynamic_blend(self, source,target,mask):
		mask_blured = self.get_blend_mask(mask)
		blend_list=[0.25,0.5,0.75,1,1,1]
		blend_ratio = blend_list[np.random.randint(len(blend_list))]
		mask_blured*=blend_ratio
		img_blended=(mask_blured * source + (1 - mask_blured) * target)
		return img_blended,mask_blured

	def get_blend_mask(self, mask):
		H,W=mask.shape
		size_h=np.random.randint(192,257)
		size_w=np.random.randint(192,257)
		mask=cv2.resize(mask,(size_w,size_h))
		kernel_1=random.randrange(5,26,2)
		kernel_1=(kernel_1,kernel_1)
		kernel_2=random.randrange(5,26,2)
		kernel_2=(kernel_2,kernel_2)
		
		mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
		mask_blured = mask_blured/(mask_blured.max())
		mask_blured[mask_blured<1]=0
		
		mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
		mask_blured = mask_blured/(mask_blured.max())
		mask_blured = cv2.resize(mask_blured,(W,H))
		return mask_blured.reshape((mask_blured.shape+(1,)))
	
	def collate_fn(self,batch):
		img_f,img_r=zip(*batch)
		data={}
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
		data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		return data
		
	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
	from albu import FrequencyPatterns,RandomDownScale

	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	parser=argparse.ArgumentParser()
	parser.add_argument('--datapath',type=str, required=True)
	args=parser.parse_args()
	datapath=args.datapath

	img_size=256
	batch_size=20
	image_dataset=selfMAD_Dataset(phase='train',image_size=img_size, datapath=datapath)

	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)
	
	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,img_size,img_size))
	utils.save_image(img, 'SelfMAD_transform_example.png', nrow=batch_size, normalize=False)
	print("image saved")

else:
	from utils.albu import FrequencyPatterns,RandomDownScale
