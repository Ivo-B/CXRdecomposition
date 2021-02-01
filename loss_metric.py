#import math
import torch
import torch.nn as nn
import numpy as np

criterion = nn.MSELoss()

def loss1(out_d, labels):
	l = (out_d - labels)
	loss_1 = 0.2 * torch.sum(torch.abs(l)) + 0.8 * torch.sqrt(torch.sum(l ** 2))
	return loss_1

def loss2(out_d, labels):
	l = (out_d - labels)
	loss_2 = torch.sqrt(torch.sum(l ** 2))
	return loss_2

def psnr(out_d, labels, data_range=255.0):
	mse = criterion(out_d, labels)
	metric = 10.0 * np.log10((data_range ** 2) / mse.item())
	return metric
