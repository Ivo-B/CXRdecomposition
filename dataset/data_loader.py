import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from volumentations import *
from generate_drr import do_full_prprocessing
# dataset paths

train_folder = "/beegfs/desy/user/ibaltrus/dataset/train"
val_folder = "/beegfs/desy/user/ibaltrus/dataset/val"
app_folder = "/beegfs/desy/user/ibaltrus/dataset/val"

class ImageData(Dataset):
	def __init__(self, train=True, input_views=1, data_aug=False):
		super(ImageData, self).__init__()
		self.train = train  # training set or test set
		self.input_views = input_views  # training set or test set
		self.transform = None
		if self.train:
			data_file = train_folder
			if data_aug:
				aug_train = []
				#return
				#aug_train.append(A.ShiftScaleRotate(p=0.2))
				#aug_train.append(
				#	A.ElasticTransform(alpha=250.0, sigma=12, alpha_affine=50, interpolation=1, border_mode=0, p=0.2))
				#aug_train.append(A.RandomBrightnessContrast(p=0.2))
				self.transform = Compose([
					#Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
					#RandomCropFromBorders(crop_value=0.1, p=0.5),
					ElasticTransform((0, 0.25), interpolation=2, p=0.2),
					#RandomDropPlane(plane_drop_prob=0.1, axes=(0, 1, 2), p=0.5),
					#Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
					#Flip(0, p=0.5),
					#Flip(1, p=0.5),
					#Flip(2, p=0.5),
					#RandomRotate90((1, 2), p=0.5),
					#GaussianNoise(var_limit=(0, 0.1), p=0.4),
					#RandomGamma(gamma_limit=(0.7, 1.5), p=0.4),
				], p=1.0)
		else:
			data_file = val_folder
		self.root_dir = data_file
		self.data = os.listdir(data_file)
		self.data = sorted(self.data)
		self.norm_max = 1.0

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img_list = os.listdir(os.path.join(self.root_dir, self.data[index]))
		img_list.sort()

		ct_image = np.load(os.path.join(self.root_dir, self.data[index], img_list[0])).astype(np.float32)
		## clip 0.5%
		p_low = np.percentile(ct_image, 0.5)
		p_high = np.percentile(ct_image, 99.5)
		# print("Clip to:", p_low, p_high)
		ct_image = np.clip(ct_image, p_low, p_high)
		# norm float data to [0, 1]
		ct_image = (ct_image - np.min(ct_image)) * (self.norm_max / (np.max(ct_image) - np.min(ct_image)))
		#print(np.min(ct_image), np.max(ct_image))
		if self.transform is not None:
			ct_image = self.transform(image=ct_image)['image'].astype(np.float32)

		# drr_front, drr_lat, drr_top = do_full_prprocessing(ct_image)
		[drr_front, drr_lat, drr_top] = do_full_prprocessing(ct_image, self.input_views)
		drr_front = (drr_front - np.min(drr_front)) * (self.norm_max / (np.max(drr_front) - np.min(drr_front)))
		if self.input_views > 1:
			drr_lat = (drr_lat - np.min(drr_lat)) * (self.norm_max / (np.max(drr_lat) - np.min(drr_lat)))
		if self.input_views > 2:
			drr_top = (drr_top - np.min(drr_top)) * (self.norm_max / (np.max(drr_top) - np.min(drr_top)))



		drr_front = (drr_front - self.norm_max)
		if self.input_views == 1:
			input_stack = torch.from_numpy(np.array([drr_front]))
		elif self.input_views == 2:
			drr_lat = (drr_lat - self.norm_max)
			input_stack = torch.from_numpy(np.array([drr_front, drr_lat]))
		elif self.input_views == 2:
			drr_lat = (drr_lat - self.norm_max)
			drr_top = (drr_top - self.norm_max)
			input_stack = torch.from_numpy(np.array([drr_front, drr_lat, drr_top]))
		else:
			raise ValueError

		ct_image = torch.from_numpy(ct_image[np.newaxis, ...])
		return input_stack, ct_image

	def __len__(self):
		return len(self.data)


class CXRDecompDataModule(pl.LightningDataModule):
	def __init__(self, batch_size, input_views=1, data_aug=False):
		super().__init__()
		self.batch_size = batch_size
		self.input_views = input_views
		self.data_aug = data_aug

	def setup(self, stage=None):
		self.cxrdecomp_train = ImageData(train=True, input_views=self.input_views, data_aug=self.data_aug)
		self.cxrdecomp_test = ImageData(train=False, input_views=self.input_views, data_aug=self.data_aug)
		self.cxrdecomp_val = ImageData(train=False, input_views=self.input_views, data_aug=self.data_aug)

	def train_dataloader(self):
		return DataLoader(self.cxrdecomp_train, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.cxrdecomp_val, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.cxrdecomp_test, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False)
