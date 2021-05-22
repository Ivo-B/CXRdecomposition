#import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from loss_metric import psnr
from ssim import SSIM
from network_aritra import CXRdecomp
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
	def __init__(self,
				 batch_size=2,
				 use_norm='none',
				 lr=0.05,
				 activation_func='relu',
				 loss_type='L2',
				 model_type='CXRdecomp2',
				 input_views=2,
				 data_range=1):
		super().__init__()
		self.loss_type = loss_type
		self.model_type = model_type
		self.input_views = input_views
		self.data_range = data_range

		self.model = CXRdecomp(batch_size,
									use_norm=use_norm,
									activation_func=activation_func,
									input_views=input_views)

		self.loss_l2 = nn.MSELoss()
		self.loss_l1 = nn.L1Loss()

		self.metric_ssim2d = SSIM(channel=1, spatial_dims=2, data_range=self.data_range)
		#self.metric_ssim3d = SSIM(channel=1, spatial_dims=3, data_range=self.data_range)
		self.save_hyperparameters()
		self.example_input_array = torch.randn((1, self.input_views, 512, 512))

	def forward(self, x):
		out, out_drr = self.model(x)
		return out, out_drr

	def training_step(self, batch, batch_idx):
		x, y = batch
		logits, logits_drr = self.model(x)
		logits = logits.to(torch.float32)
		logits_drr = logits_drr.to(torch.float32)

		self.last_logits = torch.sigmoid(logits)
		self.last_logits_drr = torch.sigmoid(logits_drr)

		loss_reco = 0.5 * torch.sqrt(self.loss_l2(logits_drr, torch.add(x, self.data_range)))
		loss_peudo_ct = 0.9 * torch.sqrt(self.loss_l2(logits, y)) + 0.1 * self.loss_l1(logits, y)
		loss = loss_peudo_ct + loss_reco

		self.log('train_loss_reco', loss_reco)
		self.log('train_loss_peudo_ct', loss_peudo_ct)

		ssim_metric = self.metric_ssim2d(self.last_logits, y)
		psnr_metric = psnr(self.last_logits, y, data_range=self.data_range)
		self.log('train_loss', loss)
		self.log('train_ssim', ssim_metric)
		self.log('train_psnr', psnr_metric)
		return loss

	def evaluate(self, batch, stage=None):
		x, y = batch
		logits, logits_drr = self.model(x)
		logits = logits.to(torch.float32)
		logits_drr = logits_drr.to(torch.float32)

		self.last_logits = torch.sigmoid(logits)
		self.last_logits_drr = torch.sigmoid(logits_drr)

		loss_reco = 0.5 * torch.sqrt(self.loss_l2(logits_drr, torch.add(x, self.data_range)))
		loss_peudo_ct = 0.9 * torch.sqrt(self.loss_l2(logits, y)) + 0.1 * self.loss_l1(logits, y)
		loss = loss_peudo_ct + loss_reco

		self.log('val_loss_reco', loss_reco)
		self.log('val_loss_peudo_ct', loss_peudo_ct)

		ssim_metric = self.metric_ssim2d(self.last_logits_val, y)
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
		gpus=4,
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