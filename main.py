#import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from loss_metric import psnr
from ssim import SSIM
from network import *
from dataset.data_loader import CXRDecompDataModule

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tb_image_callback import ImagePlotCallback

#import hydra
from omegaconf import DictConfig, OmegaConf
import os
import yaml
import argparse


#logger = logging.getLogger("lightning")
#
class LitCXRDecomp(pl.LightningModule):
	def __init__(self, batch_size=2, use_norm='none', lr=0.05, activation_func='relu', loss_type='L2',
				 model_type='CXRdecomp2', input_views=2, data_range=1):
		super().__init__()
		self.loss_type = loss_type
		self.model_type = model_type
		self.input_views = input_views
		self.data_range = data_range
		if model_type == 'Unet':
			self.model = UNet(batch_size, use_norm=use_norm, activation_func=activation_func)
		elif model_type == 'CXRdecomp1':
			self.model = CXRdecomp1(batch_size, use_norm=use_norm, activation_func=activation_func, input_views=input_views)
		elif model_type == 'CXRdecomp2':
			self.model = CXRdecomp2(batch_size,
									use_norm=use_norm,
									activation_func=activation_func,
									input_views=input_views)
		elif model_type == 'CXRdecomp3':
			self.model = CXRdecomp3(batch_size,
									use_norm=use_norm,
									activation_func=activation_func,
									input_views=input_views)
		elif model_type == 'CXRdecomp4':
			self.model = CXRdecomp4(batch_size,
									use_norm=use_norm,
									activation_func=activation_func,
									input_views=input_views)
		elif model_type == 'CXRdecomp5':
			self.model = CXRdecomp5(batch_size,
									use_norm=use_norm,
									activation_func=activation_func,
									input_views=input_views)
		elif model_type == 'CXRdecomp6':
			self.model = CXRdecomp6(batch_size,
									use_norm=use_norm,
									activation_func=activation_func,
									input_views=input_views)

		if self.loss_type == 'L2':
			self.loss = nn.MSELoss()
		elif self.loss_type == 'BCEloss':
			self.loss = nn.BCEWithLogitsLoss()
		elif self.loss_type == 'L1+L2':
			self.loss_l2 = nn.MSELoss()
			self.loss_l1 = nn.L1Loss()
		elif self.loss_type == 'L2+reco':
			self.loss_l2 = nn.MSELoss()
			# from generate_drr import do_full_prprocessing
			# import numpy as np
			# def drrLoss(logits, x):
			# 	drr_loss = 0
			# 	for i in logits.shape()[0]:
			# 		[drr_front, drr_lat, drr_top] = do_full_prprocessing(logits[i], input_views=1)
			# 		drr_front = (drr_front - np.min(drr_front)) * (1.0 / (np.max(drr_front) - np.min(drr_front)))
			# 		drr_front = torch.Tensor((drr_front * 2 - 1))
			# 		drr_loss += nn.MSELoss(drr_front, [x])
		self.metric_ssim2d = SSIM(channel=1, spatial_dims=2, data_range=self.data_range)
		self.metric_ssim3d = SSIM(channel=1, spatial_dims=3, data_range=self.data_range)
		self.save_hyperparameters()
		self.example_input_array = torch.randn((1, self.input_views, 256, 256))

	def forward(self, x):
		if self.model_type == 'CXRdecomp2':
			out, out_drr = self.model(x)
			return out, out_drr
		else:
			out = self.model(x)
			return out

	def training_step(self, batch, batch_idx):
		x, y = batch
		if self.model_type == 'CXRdecomp2':
			logits, logits_drr = self.model(x)
			logits = logits.to(torch.float32)
			logits_drr = logits_drr.to(torch.float32)

			self.last_logits = torch.sigmoid(logits)
			self.last_logits_drr = torch.sigmoid(logits_drr)

			loss_reco = 0.2 * self.loss(logits_drr, torch.add(x, self.data_range))
			loss_peudo_ct = 0.8 * self.loss(logits, y)
			loss = loss_peudo_ct + loss_reco

			self.log('train_loss_reco', loss_reco)
			self.log('train_loss_peudo_ct', loss_peudo_ct)

		else:
			logits = self.model(x).to(torch.float32)

			if self.loss_type == 'BCEloss':
				self.last_logits = torch.sigmoid(logits)
			else:
				self.last_logits = logits

			if self.loss_type == 'L1+L2':
				loss = 0.2 * self.loss_l1(logits, y) + 0.8 * self.loss_l2(logits, y)
			else:
				loss = self.loss(logits, y)
		ssim_metric = self.metric_ssim3d(self.last_logits, y)
		psnr_metric = psnr(self.last_logits, y, data_range=self.data_range)
		self.log('train_loss', loss)
		self.log('train_ssim', ssim_metric)
		self.log('train_psnr', psnr_metric)
		return loss

	def evaluate(self, batch, stage=None):
		x, y = batch
		if self.model_type == 'CXRdecomp2':
			logits, logits_drr = self(x)
			logits = logits.to(torch.float32)
			logits_drr = logits_drr.to(torch.float32)


			self.last_logits_val = torch.sigmoid(logits)
			self.last_logits_drr_val = torch.sigmoid(logits_drr)

			loss_reco = 0.2 * self.loss(logits_drr, torch.add(x, self.data_range))
			loss_peudo_ct = 0.8 * self.loss(logits, y)
			loss = loss_peudo_ct + loss_reco

			self.log('val_loss_reco', loss_reco)
			self.log('val_loss_peudo_ct', loss_peudo_ct)
		else:
			logits = self(x).to(torch.float32)
			if self.loss_type == 'BCEloss':
				self.last_logits_val = torch.sigmoid(logits)
			else:
				self.last_logits_val = logits
			if self.loss_type == 'L1+L2':
				loss = 0.2 * self.loss_l1(logits, y) + 0.8 * self.loss_l2(logits, y)
			else:
				loss = self.loss(logits, y)
		ssim_metric = self.metric_ssim3d(self.last_logits_val, y)
		psnr_metric = psnr(self.last_logits_val, y, data_range=self.data_range)

		if stage:
			self.log(f'{stage}_loss', loss, prog_bar=True)
			self.log(f'{stage}_ssim', ssim_metric, prog_bar=True)
			self.log(f'{stage}_psnr', psnr_metric, prog_bar=False)

	def validation_step(self, batch, batch_idx):
		self.evaluate(batch, 'val')

	def test_step(self, batch, batch_idx):
		self.evaluate(batch, 'test')

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)#, weight_decay=5e-4)
		steps_per_epoch = 146 // self.hparams.batch_size
		scheduler_dict = {
			#'scheduler': OneCycleLR(optimizer, 0.001, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch),
			#'interval': 'step'
			'scheduler': ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30),
			'monitor': 'val_loss'
		}
		return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


#@hydra.main(config_path="configs")
def main(config_path) :
	#logger = logging.getLogger(__name__)
	# data loading
	os.chdir('/beegfs/desy/user/ibaltrus/repos/cxrDecomp')
	with open(config_path) as file:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to Python the dictionary format
		cfg = yaml.load(file, Loader=yaml.FullLoader)
	cfg = OmegaConf.create(cfg)
	print(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
	if cfg.general.overfit_batches > 0:
		print(f"Start overfitting test with {cfg.general.overfit_batches} batches!")

	if cfg.general.overfit_batches > 0:
		tb_log_dir = '/beegfs/desy/user/ibaltrus/repos/cxrDecomp/lightning_logs/cxrdecomp_overfitting'
	else:
		tb_log_dir = '/beegfs/desy/user/ibaltrus/repos/cxrDecomp/lightning_logs/cxrdecomp_experiment'
	if not os.path.isdir(tb_log_dir):
		os.makedirs(tb_log_dir)

	cxrdecomp_dm = CXRDecompDataModule(cfg.general.batch_size, input_views=cfg.network.input_views, data_aug=cfg.general.data_aug)

	model = LitCXRDecomp(batch_size=cfg.general.batch_size, lr=cfg.general.learning_rate,
						 loss_type=cfg.general.loss, model_type=cfg.network.model,
						 activation_func=cfg.network.activation_func,
						 use_norm=cfg.network.use_norm, input_views=cfg.network.input_views)
	model.datamodule = cxrdecomp_dm

	checkpoint_callback = ModelCheckpoint(
		monitor='val_loss',
		save_last=True,
		filename='cxrdecomp-{epoch:02d}-{val_loss:.2f}',
		save_top_k=3,
		mode='min',
	)
	early_stop_callback = EarlyStopping(
		monitor='val_loss',
		min_delta=0.00,
		patience=50,
		verbose=False,
		mode='min'
	)

	trainer = pl.Trainer(
		accelerator='ddp',
		#accumulate_grad_batches=16,
		progress_bar_refresh_rate=2,
		max_epochs=1000,
		gpus=2,
		auto_select_gpus=True,
		logger=loggers.TensorBoardLogger(tb_log_dir, name=f'{os.path.splitext(os.path.basename(config_path))[0]}', log_graph=True),
		callbacks=[LearningRateMonitor(logging_interval='step'), ImagePlotCallback(), checkpoint_callback, early_stop_callback],
		precision=16,
		weights_summary='full',
		overfit_batches=cfg.general.overfit_batches,
		#gradient_clip_val=2
	)

	trainer.fit(model, cxrdecomp_dm)
	return trainer

if __name__ == '__main__':
	pl.seed_everything(42)
	parser = argparse.ArgumentParser(
		description="Evaluates a trained model given the root path")

	parser.add_argument('--config', type=str,
						default='/beegfs/desy/user/ibaltrus/repos/cxrDecomp/configs/CXRdecomp3.yaml',
						help='Path to a config model which is to be tested')

	args = parser.parse_args()
	args_dict = vars(args)
	main(args_dict['config'])

### TODO:
# done training with frontral + lateral; L2-loss -> run number version_15
# done training with L1+L2 loss -> reduce blurry images ?! - >version_16
# currently training with BCE loss -> reduce blurry images ?! - >version_17
# currently training single_view with L2 loss and drrLoss -> reduce blurry images ?! - >version_18
# train with only fontral oder lateral images
# train with 3d input where lateral and frontal are in the middle slice -> new architecture