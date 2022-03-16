from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from utils.mish.mish import Mish


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
	"""3x3 convolution with padding"""
	return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
	"""1x1 convolution"""
	return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

	expansion: int = 4

	def __init__(
			self,
			inplanes: int,
			planes: int,
			stride: int = 1,
			downsample: Optional[nn.Module] = None,
			groups: int = 1,
			base_width: int = 64,
			dilation: int = 1,
			norm_layer: Optional[Callable[..., nn.Module]] = None,
			dim: str = '2d',
			use_norm: str = 'none',
			activation_func: str = 'relu'
	) -> None:
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			if dim == '2d':
				norm_layer = nn.BatchNorm2d
			else:
				norm_layer = nn.BatchNorm3d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		if dim == '2d':
			conv_layer1 = conv1x1
			conv_layer3 = conv3x3
		else:
			conv_layer1 = conv1x1x1
			conv_layer3 = conv3x3x3
		if activation_func == 'relu':
			activation_func = nn.ReLU
		else:
			activation_func = Mish

		self.conv1 = conv_layer1(inplanes, width)
		if use_norm == 'group_norm':
			self.bn1 = norm_layer(2, width)
		else:
			self.bn1 = norm_layer(width)
		if activation_func.__name__ == 'ReLu':
			self.act_func1 = activation_func(inplace=True)
		else:
			self.act_func1 = activation_func()

		self.conv2 = conv_layer3(width, width, stride, groups, dilation)
		if use_norm == 'group_norm':
			self.bn2 = norm_layer(2, width)
		else:
			self.bn2 = norm_layer(width)
		if activation_func.__name__ == 'ReLu':
			self.act_func2 = activation_func(inplace=True)
		else:
			self.act_func2 = activation_func()

		self.conv3 = conv_layer1(width, planes * self.expansion)
		if use_norm == 'group_norm':
			self.bn3 = norm_layer(2, planes * self.expansion)
		else:
			self.bn3 = norm_layer(planes * self.expansion)
		if activation_func.__name__ == 'ReLu':
			self.act_func3 = activation_func(inplace=True)
		else:
			self.act_func3 = activation_func()

		if use_norm == 'group_norm':
			self.downsample = nn.Sequential(
				conv_layer1(inplanes, planes * self.expansion, stride),
				norm_layer(2, planes * self.expansion),
			)
		else:
			self.downsample = nn.Sequential(
				conv_layer1(inplanes, planes * self.expansion, stride),
				norm_layer(planes * self.expansion),
			)
		self.stride = stride


	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act_func1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.act_func2(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.act_func3(out)

		return out


class ChannelSELayer(nn.Module):
	"""
	Re-implementation of Squeeze-and-Excitation (SE) block described in:
		*Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
	"""

	def __init__(self, num_channels, reduction_ratio=1, activation_func: str = 'relu'):
		"""
		:param num_channels: No of input channels
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(ChannelSELayer, self).__init__()
		num_channels_reduced = num_channels // reduction_ratio
		self.reduction_ratio = reduction_ratio
		self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
		self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
		if activation_func == 'relu':
			self.act_func = nn.ReLU(inplace=True)
		else:
			self.act_func = Mish()
		#self.relu = Mish()
		self.sigmoid = nn.Sigmoid()

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, H, W)
		:return: output tensor
		"""
		batch_size, num_channels, H, W = input_tensor.size()
		# Average along each channel
		squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

		# channel excitation
		fc_out_1 = self.act_func(self.fc1(squeeze_tensor))
		fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

		a, b = squeeze_tensor.size()
		output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
		return output_tensor


class ChannelSELayer3D(nn.Module):
	"""
	3D extension of Squeeze-and-Excitation (SE) block described in:
		*Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
		*Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
	"""

	def __init__(self, num_channels, reduction_ratio=2, activation_func: str = 'relu'):
		"""
		:param num_channels: No of input channels
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(ChannelSELayer3D, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		num_channels_reduced = num_channels // reduction_ratio
		self.reduction_ratio = reduction_ratio
		self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
		self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
		if activation_func == 'relu':
			self.act_func = nn.ReLU(inplace=True)
		else:
			self.act_func = Mish()

		self.sigmoid = nn.Sigmoid()

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
		:return: output tensor
		"""
		batch_size, num_channels, D, H, W = input_tensor.size()
		# Average along each channel
		squeeze_tensor = self.avg_pool(input_tensor)

		# channel excitation
		fc_out_1 = self.act_func(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
		fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

		output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

		return output_tensor


class UNet(nn.Module):

	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu'):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'batch_norm':
			norm_layer2d = nn.BatchNorm2d
			norm_layer3d = nn.BatchNorm3d
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		if activation_func == 'relu':
			activ_func = nn.ReLU
		else:
			activ_func = Mish

		self.batch_size = batch_size
		self.conv_1 = nn.Conv2d(2, 256, (3, 3), padding=1)
		if use_norm == 'group_norm':
			self.bn_1 = norm_layer2d(2, 256)
		else:
			self.bn_1 = norm_layer2d(256)
		if activ_func.__name__ == 'ReLU':
			self.activ_func1 = activ_func(inplace=True)
		else:
			self.activ_func1 = activ_func()
		self.dconv_down1 = Bottleneck(256, 64, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool1 = nn.MaxPool2d(2)
		self.dconv_down2 = Bottleneck(256, 128, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool2 = nn.MaxPool2d(2)
		self.dconv_down3 = Bottleneck(512, 256, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool3 = nn.MaxPool2d(2)
		self.dconv_down4 = Bottleneck(1024, 512, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool4 = nn.MaxPool2d(2)
		self.dconv_down5 = Bottleneck(2048, 1024, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool5 = nn.MaxPool2d(2)

		# self.se_trans1 = ChannelSELayer(num_channels=4096, reduction_ratio=2)
		self.conv_trans1 = nn.Conv2d(4096, 4096, (1, 1), padding=0)
		if use_norm == 'group_norm':
			self.bn_trans1 = norm_layer2d(2, 4096)
		else:
			self.bn_trans1 = norm_layer2d(4096)
		if activ_func.__name__ == 'ReLU':
			self.act_func1 = activ_func(inplace=True)
		else:
			self.act_func1 = activ_func()

		# self.se_trans2 = ChannelSELayer3D(num_channels=2048, reduction_ratio=2)
		self.conv_trans2 = nn.Conv3d(2048, 2048, (1, 1, 1), padding=0)
		if use_norm == 'group_norm':
			self.bn_trans2 = norm_layer3d(2, 2048)
		else:
			self.bn_trans2 = norm_layer3d(2048)
		if activ_func.__name__ == 'ReLU':
			self.act_func2 = activ_func(inplace=True)
		else:
			self.act_func2 = activ_func()

		# self.se_trans3 = ChannelSELayer3D(num_channels=1024, reduction_ratio=2)
		self.conv_trans3 = nn.Conv3d(1024, 1024, (1, 1, 1), padding=0)
		if use_norm == 'group_norm':
			self.bn_trans3 = norm_layer3d(2, 1024)
		else:
			self.bn_trans3 = norm_layer3d(1024)
		if activ_func.__name__ == 'ReLU':
			self.act_func3 = activ_func(inplace=True)
		else:
			self.act_func3 = activ_func()

		# self.block_trans1 = Bottleneck(4096, 1024, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)
		#
		# self.conv_trans2 = nn.Conv3d(2048, 2048, (1, 1, 1), padding=0)
		# if use_norm == 'group_norm':
		# 	self.bn_trans2 = norm_layer3d(2, 2048)
		# else:
		# 	self.bn_trans2 = norm_layer3d(2048)
		# if activ_func.__name__ == 'ReLU':
		# 	self.act_func2 = activ_func(inplace=True)
		# else:
		# 	self.act_func2 = activ_func()
		self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.block_trans3 = Bottleneck(1024, 256, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d', activation_func=activation_func)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.econv_up1 = Bottleneck(512, 64, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d', activation_func=activation_func)
		self.econv_up2 = Bottleneck(256, 32, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d', activation_func=activation_func)
		self.econv_up3 = Bottleneck(128, 16, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d', activation_func=activation_func)
		self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.econv_up4 = Bottleneck(64, 8, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d', activation_func=activation_func)
		self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.econv_out = nn.Conv3d(32, 1, (1, 1, 1), padding=0)
		self.upsample5 = nn.Upsample(scale_factor=2, mode='trilinear')

	def forward(self, x_1):
		#print(x_1.shape)
		x_1 = self.conv_1(x_1)
		x_1 = self.bn_1(x_1)
		x_1 = self.activ_func1(x_1)
		conv1 = self.dconv_down1(x_1)  # 256, 256 ; 3 -> 256
		x = self.maxpool1(conv1)  # 128, 128 ; 256

		#print(x.shape)
		conv2 = self.dconv_down2(x)  # 128, 128 ; 256 -> 512
		x = self.maxpool2(conv2)  # 64, 64 ; 512

		#print(x.shape)
		conv3 = self.dconv_down3(x)  # 64, 64 ; 512 -> 1024
		x = self.maxpool3(conv3)  # 32, 32 ; 1024

		conv4 = self.dconv_down4(x)  # 32, 32 ; 1024 -> 2048
		x = self.maxpool4(conv4)  # 16, 16; 256

		conv5 = self.dconv_down5(x)  # 16, 16 ; 2048 -> 4096
		x = self.maxpool5(conv5)  # 8, 8; 4096

		#### old trans
		#x = self.se_trans1(x)
		x = self.conv_trans1(x)
		x = self.bn_trans1(x)
		x = self.act_func1(x)

		x = x.reshape((x.shape[0], 2048, 2, 8, 8))
		#x = self.se_trans2(x)
		x = self.conv_trans2(x)
		x = self.bn_trans2(x)
		x = self.act_func2(x)
		x = self.upsample1(x)

		x = x.reshape((x.shape[0], 1024, 8, 16, 16))
		#x = self.se_trans3(x)
		x = self.conv_trans3(x)
		x = self.bn_trans3(x)
		x = self.act_func3(x)
		x = self.upsample2(x)

		### new trans
		# x = self.block_trans1(x)
		#
		# x = x.reshape((x.shape[0], 2048, 2, 8, 8))
		# x = self.conv_trans2(x)
		# x = self.bn_trans2(x)
		# x = self.act_func2(x)
		# x = self.upsample1(x)
		#
		# x = x.reshape((x.shape[0], 1024, 8, 16, 16))
		# x = self.block_trans3(x)
		# x = self.upsample2(x)

		x = x.reshape((x.shape[0], 512, 32, 32, 32))
		x = self.econv_up1(x) # 512 -> 256; 32 x 32 x 32
		x = self.upsample3(x) # 64 x 64 x 64

		x = self.econv_up2(x)  # 256 -> 128; 64 x 64 64
		x = self.upsample4(x)  # 128 x 128 x 128

		x = self.econv_up3(x)  # 128 -> 64; 128 x 128 x 128
		x = self.upsample5(x)  # 256 x 256 x 256

		x = self.econv_up4(x)  # 64 -> 32; 256 x 256 x 256
		x = self.econv_out(x)  # 32 -> 1; 256 x 256 x 256
		out_d = x

		return out_d


class CXRdecomp1(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		if activation_func == 'relu':
			activ_func = nn.ReLU
		else:
			activ_func = Mish

		self.batch_size = batch_size
		self.conv_1 = nn.Conv2d(input_views, 256, (3, 3), padding=1)
		if use_norm == 'group_norm':
			self.bn_1 = norm_layer2d(2, 256)
		else:
			self.bn_1 = norm_layer2d
		if activ_func.__name__ == 'ReLU':
			self.activ_func1 = activ_func(inplace=True)
		else:
			self.activ_func1 = activ_func()

		self.dconv_down1 = Bottleneck(256, 64, use_norm=use_norm, norm_layer=norm_layer2d,
									  activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool1 = nn.MaxPool2d(2)
		self.dconv_down2 = Bottleneck(256, 128, use_norm=use_norm, norm_layer=norm_layer2d,
									  activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool2 = nn.MaxPool2d(2)
		self.dconv_down3 = Bottleneck(512, 256, use_norm=use_norm, norm_layer=norm_layer2d,
									  activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool3 = nn.MaxPool2d(2)
		self.dconv_down4 = Bottleneck(1024, 512, use_norm=use_norm, norm_layer=norm_layer2d,
									  activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool4 = nn.MaxPool2d(2)
		self.dconv_down5 = Bottleneck(2048, 1024, use_norm=use_norm, norm_layer=norm_layer2d,
									  activation_func=activation_func)  # 256, 256 ; 3 -> 32
		self.maxpool5 = nn.MaxPool2d(2)

		# self.se_trans1 = ChannelSELayer(num_channels=4096, reduction_ratio=2)
		self.conv_trans1 = nn.Conv2d(4096, 4096, (1, 1), padding=0)
		if use_norm == 'group_norm':
			self.bn_trans1 = norm_layer2d(2, 4096)
		else:
			self.bn_trans1 = norm_layer2d(4096)
		if activ_func.__name__ == 'ReLU':
			self.act_func1 = activ_func(inplace=True)
		else:
			self.act_func1 = activ_func()

		# self.se_trans2 = ChannelSELayer3D(num_channels=2048, reduction_ratio=2)
		self.conv_trans2 = nn.Conv3d(2048, 2048, (1, 1, 1), padding=0)
		if use_norm == 'group_norm':
			self.bn_trans2 = norm_layer3d(2, 2048)
		else:
			self.bn_trans2 = norm_layer3d(2048)
		if activ_func.__name__ == 'ReLU':
			self.act_func2 = activ_func(inplace=True)
		else:
			self.act_func2 = activ_func()

		# self.se_trans3 = ChannelSELayer3D(num_channels=1024, reduction_ratio=2)
		self.conv_trans3 = nn.Conv3d(1024, 1024, (1, 1, 1), padding=0)
		if use_norm == 'group_norm':
			self.bn_trans3 = norm_layer3d(2, 1024)
		else:
			self.bn_trans3 = norm_layer3d(1024)
		if activ_func.__name__ == 'ReLU':
			self.act_func3 = activ_func(inplace=True)
		else:
			self.act_func3 = activ_func()

		# self.block_trans1 = Bottleneck(4096, 1024, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)
		#
		# self.conv_trans2 = nn.Conv3d(2048, 2048, (1, 1, 1), padding=0)
		# if use_norm == 'group_norm':
		# 	self.bn_trans2 = norm_layer3d(2, 2048)
		# else:
		# 	self.bn_trans2 = norm_layer3d(2048)
		# if activ_func.__name__ == 'ReLU':
		# 	self.act_func2 = activ_func(inplace=True)
		# else:
		# 	self.act_func2 = activ_func()
		self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.block_trans3 = Bottleneck(1024, 256, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d',
									   activation_func=activation_func)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.econv_up1 = Bottleneck(512, 64, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d',
									activation_func=activation_func)
		self.econv_up2 = Bottleneck(256, 32, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d',
									activation_func=activation_func)
		self.econv_up3 = Bottleneck(128, 16, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d',
									activation_func=activation_func)
		self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.econv_up4 = Bottleneck(64, 8, use_norm=use_norm, norm_layer=norm_layer3d, dim='3d',
									activation_func=activation_func)
		self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.upsample5 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.econv_out = nn.Conv3d(32, 1, (1, 1, 1), padding=0)

		self.econv_out_drr = nn.Conv2d(256, 128, (1, 1), padding=0)
		if use_norm == 'group_norm':
			self.ebn_out_drr = norm_layer2d(2, 128)
		else:
			self.ebn_out_drr = norm_layer2d(128)
		if activ_func.__name__ == 'ReLU':
			self.eact_out_drr = activ_func(inplace=True)
		else:
			self.eact_out_drr = activ_func()

		self.econv2_out_drr = nn.Conv2d(128, 64, (1, 1), padding=0)
		if use_norm == 'group_norm':
			self.ebn2_out_drr = norm_layer2d(2, 64)
		else:
			self.ebn2_out_drr = norm_layer2d(64)
		if activ_func.__name__ == 'ReLU':
			self.eact2_out_drr = activ_func(inplace=True)
		else:
			self.eact2_out_drr = activ_func()

		self.econv3_out_drr = nn.Conv2d(64, input_views, (1, 1), padding=0)

	def forward(self, x_1):
		# print(x_1.shape)
		x_1 = self.conv_1(x_1)
		x_1 = self.bn_1(x_1)
		x_1 = self.activ_func1(x_1)
		conv1 = self.dconv_down1(x_1)  # 256, 256 ; 3 -> 256
		x = self.maxpool1(conv1)  # 128, 128 ; 256

		# print(x.shape)
		conv2 = self.dconv_down2(x)  # 128, 128 ; 256 -> 512
		x = self.maxpool2(conv2)  # 64, 64 ; 512

		# print(x.shape)
		conv3 = self.dconv_down3(x)  # 64, 64 ; 512 -> 1024
		x = self.maxpool3(conv3)  # 32, 32 ; 1024

		conv4 = self.dconv_down4(x)  # 32, 32 ; 1024 -> 2048
		x = self.maxpool4(conv4)  # 16, 16; 256

		conv5 = self.dconv_down5(x)  # 16, 16 ; 2048 -> 4096
		x = self.maxpool5(conv5)  # 8, 8; 4096

		#### old trans
		# x = self.se_trans1(x)
		x = self.conv_trans1(x)
		x = self.bn_trans1(x)
		x = self.act_func1(x)

		x = x.reshape((x.shape[0], 2048, 2, 8, 8))
		# x = self.se_trans2(x)
		x = self.conv_trans2(x)
		x = self.bn_trans2(x)
		x = self.act_func2(x)
		x = self.upsample1(x)

		x = x.reshape((x.shape[0], 1024, 8, 16, 16))
		# x = self.se_trans3(x)
		x = self.conv_trans3(x)
		x = self.bn_trans3(x)
		x = self.act_func3(x)
		x = self.upsample2(x)

		### new trans
		# x = self.block_trans1(x)
		#
		# x = x.reshape((x.shape[0], 2048, 2, 8, 8))
		# x = self.conv_trans2(x)
		# x = self.bn_trans2(x)
		# x = self.act_func2(x)
		# x = self.upsample1(x)
		#
		# x = x.reshape((x.shape[0], 1024, 8, 16, 16))
		# x = self.block_trans3(x)
		# x = self.upsample2(x)

		x = x.reshape((x.shape[0], 512, 32, 32, 32))
		x = self.econv_up1(x)  # 512 -> 256; 32 x 32 x 32
		x = self.upsample3(x)  # 64 x 64 x 64

		x = self.econv_up2(x)  # 256 -> 128; 64 x 64 64
		x = self.upsample4(x)  # 128 x 128 x 128

		x = self.econv_up3(x)  # 128 -> 64; 128 x 128 x 128
		x = self.upsample5(x)  # 256 x 256 x 256

		x = self.econv_up4(x)  # 64 -> 32; 256 x 256 x 256
		x = self.econv_out(x)  # 32 -> 1; 256 x 256 x 256
		out_d = x
		x = torch.squeeze(x, 1)
		x = self.econv_out_drr(x)
		x = self.ebn_out_drr(x)
		x = self.eact_out_drr(x)

		x = self.econv2_out_drr(x)
		x = self.ebn2_out_drr(x)
		x = self.eact2_out_drr(x)

		out_drr = self.econv3_out_drr(x)

		return out_d, out_drr


class ConvNormActiv(nn.Module):
	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			norm_layer: Optional[Callable[..., nn.Module]] = None,
			use_norm: str = 'none',
			dim: str = '2d',
			kernel_size: int = 3,
			activation_func: str = 'relu'
	) -> None:
		super(ConvNormActiv, self).__init__()
		if dim == '2d':
			if kernel_size != 3:
				conv_layer = conv1x1
			else:
				conv_layer = conv3x3
		else:
			if kernel_size != 3:
				conv_layer = conv1x1x1
			else:
				conv_layer = conv3x3x3
		if activation_func == 'relu':
			activation_func = nn.ReLU
		elif activation_func == 'leakyRelu':
			activation_func = nn.LeakyReLU
		elif activation_func == 'mish':
			activation_func = Mish
		else:
			activation_func = nn.Identity

		self.conv1 = conv_layer(in_channels, out_channels)
		if use_norm == 'group_norm':
			self.bn1 = norm_layer(2, out_channels)
		else:
			self.bn1 = norm_layer(out_channels)
		if activation_func.__name__ == 'ReLu':
			self.act_func1 = activation_func(inplace=True)
		elif activation_func == 'leakyReLU':
			self.act_func1 = activation_func(inplace=True)
		else:
			self.act_func1 = activation_func()

	def forward(self, x: Tensor) -> Tensor:
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act_func1(out)
		return out


class ResidualBlock(nn.Module):

	expansion: int = 4

	def __init__(
			self,
			in_channels: int,
			out_channels: int,
			stride: int = 1,
			norm_layer: Optional[Callable[..., nn.Module]] = None,
			use_norm: str = 'none',
			dim: str = '2d',
			kernel_size: int = 3,
			activation_func: str = 'relu',
			downsample: bool = False
	) -> None:
		super(ResidualBlock, self).__init__()
		if dim == '2d':
			if kernel_size != 3:
				conv_layer = conv1x1
			else:
				conv_layer = conv3x3
		else:
			if kernel_size != 3:
				conv_layer = conv1x1x1
			else:
				conv_layer = conv3x3x3
		if activation_func == 'relu':
			activation_func = nn.ReLU
		elif activation_func == 'leakyRelu':
			activation_func = nn.LeakyReLU
		else:
			activation_func = Mish
		self.downsample = downsample
		self.in_channels = in_channels
		self.out_channels = out_channels
		if downsample:
			self.conv_down = conv_layer(in_channels, out_channels, stride=2)
			if use_norm == 'group_norm':
				self.bn_down = norm_layer(2, out_channels)
			else:
				self.bn_down = norm_layer(out_channels)
			if activation_func.__name__ == 'ReLu':
				self.act_down = activation_func(inplace=True)
			elif activation_func == 'leakyRelu':
				self.act_down = activation_func(inplace=True)
			else:
				self.act_down = activation_func()

		if not downsample:
			self.conv1 = conv_layer(in_channels, out_channels)
			if use_norm == 'group_norm':
				self.bn1 = norm_layer(2, out_channels)
			else:
				self.bn1 = norm_layer(out_channels)
			if activation_func.__name__ == 'ReLu':
				self.act_func1 = activation_func(inplace=True)
			elif activation_func == 'leakyRelu':
				self.act_func1 = activation_func(inplace=True)
			else:
				self.act_func1 = activation_func()


			self.conv2 = conv_layer(out_channels, out_channels)
			if use_norm == 'group_norm':
				self.bn2 = norm_layer(2, out_channels)
			else:
				self.bn2 = norm_layer(out_channels)
			if activation_func.__name__ == 'ReLu':
				self.act_func2 = activation_func(inplace=True)
			elif activation_func == 'leakyRelu':
				self.act_func2 = activation_func(inplace=True)
			else:
				self.act_func2 = activation_func()

			if self.in_channels != self.out_channels:
				if dim == '2d':
					conv_layer1 = conv1x1
				else:
					conv_layer1 = conv1x1x1

				if use_norm == 'group_norm':
					self.upsample_channels = nn.Sequential(
						conv_layer1(in_channels, out_channels),
						norm_layer(2, out_channels),
					)
				else:
					self.upsample_channels = nn.Sequential(
						conv_layer1(in_channels, out_channels),
						norm_layer(out_channels),
					)
		else:
			self.conv1 = conv_layer(in_channels, in_channels)
			if use_norm == 'group_norm':
				self.bn1 = norm_layer(2, in_channels)
			else:
				self.bn1 = norm_layer(in_channels)
			if activation_func.__name__ == 'ReLu':
				self.act_func1 = activation_func(inplace=True)
			elif activation_func == 'leakyRelu':
				self.act_func1 = activation_func(inplace=True)
			else:
				self.act_func1 = activation_func()

			self.conv2 = conv_layer(in_channels, in_channels)
			if use_norm == 'group_norm':
				self.bn2 = norm_layer(2, in_channels)
			else:
				self.bn2 = norm_layer(in_channels)
			if activation_func.__name__ == 'ReLu':
				self.act_func2 = activation_func(inplace=True)
			elif activation_func == 'leakyRelu':
				self.act_func2 = activation_func(inplace=True)
			else:
				self.act_func2 = activation_func()



	def forward(self, x: Tensor) -> Tensor:

		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act_func1(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if not self.downsample:
			if self.in_channels != self.out_channels:
				identity = self.upsample_channels(identity)

		out += identity
		out = self.act_func2(out)

		if self.downsample:
			out = self.conv_down(out)
			out = self.bn_down(out)
			out = self.act_down(out)

		return out


class CXRdecomp2(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		self.batch_size = batch_size
		self.input_views = input_views
		self.cna1 = ConvNormActiv(input_views, 128, activation_func=activation_func, norm_layer=norm_layer2d,
										use_norm=use_norm, kernel_size=3, dim='2d')
		self.enc_cna1 = ResidualBlock(128, 128, activation_func=activation_func, norm_layer=norm_layer2d, use_norm=use_norm, kernel_size=3, downsample=True)
		self.enc_cna2 = ResidualBlock(128, 256, activation_func=activation_func, norm_layer=norm_layer2d, use_norm=use_norm, kernel_size=3, downsample=True) # 128
		self.enc_cna3 = ResidualBlock(256, 512, activation_func=activation_func, norm_layer=norm_layer2d, use_norm=use_norm, kernel_size=3, downsample=True) # 64
		self.enc_cna4 = ResidualBlock(512, 1024, activation_func=activation_func, norm_layer=norm_layer2d, use_norm=use_norm, kernel_size=3, downsample=True) # 32
		self.enc_cna5 = ResidualBlock(1024, 2048, activation_func=activation_func, norm_layer=norm_layer2d, use_norm=use_norm, kernel_size=3, downsample=True) # 16

		self.trans_cna1 = ConvNormActiv(1024, 1024, activation_func=activation_func, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')
		self.trans_upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.trans_cna2 = ConvNormActiv(512, 512, activation_func=activation_func, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')
		self.trans_upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.trans_cna3 = ConvNormActiv(256, 256, activation_func=activation_func, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')
		self.trans_upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.dec_cna1 = ResidualBlock(256, 128, activation_func=activation_func, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')
		self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.dec_cna2 = ResidualBlock(128, 64, activation_func=activation_func, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')
		self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.dec_cna3 = ResidualBlock(64, 32, activation_func=activation_func, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')
		self.dec_cna4 = ConvNormActiv(32, 1, activation_func=None, norm_layer=norm_layer3d, use_norm=use_norm, kernel_size=3, dim='3d')

	def forward(self, x_in):
		x = self.cna1(x_in)
		x = self.enc_cna1(x)
		x = self.enc_cna2(x)
		x = self.enc_cna3(x)
		x = self.enc_cna4(x)
		x = self.enc_cna5(x)

		x = x.reshape((x.shape[0], 1024, 2, 8, 8))
		x = self.trans_cna1(x)
		x = self.trans_upsample1(x) # 1024, 4 ,16, 16

		x = x.reshape((x.shape[0], 512, 8, 16, 16))
		x = self.trans_cna2(x)
		x = self.trans_upsample2(x)  # 512, 16 ,32, 32

		x = x.reshape((x.shape[0], 256, 32, 32, 32))
		x = self.trans_cna3(x)
		x = self.trans_upsample3(x)  # 256, 64 ,64, 64

		x = self.dec_cna1(x) # 128, 64 ,64, 64
		x = self.dec_upsample1(x) # 128, 128 ,128, 128

		x = self.dec_cna2(x)  # 64, 128 ,128, 128
		x = self.dec_upsample2(x)  # 64, 256 ,256, 256

		x = self.dec_cna3(x)  # 32, 256 ,256, 256
		out_y = self.dec_cna4(x) # 1, 256 ,256, 256

		#out_y = torch.relu(out_y)
		# remove dim with 1
		x = out_y.reshape((out_y.shape[0], 256, 256, 256))

		# drr projection for frontal
		x = torch.div(torch.add(torch.mul(x, 0.2), 1000.), (x.shape[1] * 1000))
		out_drr = torch.exp(torch.add(torch.sum(x, dim=[1], keepdim=True), 0.02))

		out_drr = (out_drr - torch.min(out_drr)) * (1. / ((torch.max(out_drr) - torch.min(out_drr)) + 1e-5))
		if self.input_views > 1:
			out_drr_lat = torch.unsqueeze(torch.exp(torch.add(torch.sum(x, dim=[3], keepdim=False), 0.02)), dim=1)
			out_drr_lat = torch.transpose(out_drr_lat, 2, 3)
			out_drr = torch.cat((out_drr, out_drr_lat), dim=1)

		return out_y, out_drr


class DoubleConvBlock2d(nn.Module):
	def __init__(self, num_channels, num_channels_out, padding=1, stride=1, use_norm='group_norm', activation_func: str = 'relu'):
		"""
		:param num_channels: No of input channels
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(DoubleConvBlock2d, self).__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm

		if activation_func == 'relu':
			activ_func = nn.ReLU
		elif activation_func == 'leakyRelu':
			activ_func = nn.LeakyReLU
		else:
			activ_func = Mish

		self.conv_1 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=padding, stride=1)
		self.bn_1 = norm_layer2d(2, num_channels)
		if activ_func.__name__ == 'ReLU' or activ_func.__name__ == 'leakyRelu':
			self.activ_func1 = activ_func(inplace=True)
		else:
			self.activ_func1 = activ_func()

		self.conv_2 = nn.Conv2d(num_channels, num_channels_out, (3, 3), padding=padding, stride=stride)
		self.bn_2 = norm_layer2d(2, num_channels_out)
		if activ_func.__name__ == 'ReLU' or activ_func.__name__ == 'leakyRelu':
			self.activ_func2 = activ_func(inplace=True)
		else:
			self.activ_func2 = activ_func()

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
		:return: output tensor
		"""
		x = self.conv_1(input_tensor)
		x = self.bn_1(x)
		x = self.activ_func1(x)

		x = self.conv_2(x)
		x = self.bn_2(x)
		x = self.activ_func2(x)

		return x

class DoubleConvBlock3d(nn.Module):
	def __init__(self, num_channels, num_channels_out, padding=1, stride=1, use_norm='group_norm', activation_func: str = 'relu'):
		"""
		:param num_channels: No of input channels
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(DoubleConvBlock3d, self).__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm

		if activation_func == 'relu':
			activ_func = nn.ReLU
		elif activation_func == 'leakyRelu':
			activ_func = nn.LeakyReLU
		else:
			activ_func = Mish

		self.conv_1 = nn.Conv3d(num_channels, num_channels, (3, 3, 3), padding=padding, stride=1)
		self.bn_1 = norm_layer2d(2, num_channels)
		if activ_func.__name__ == 'ReLU' or activ_func.__name__ == 'leakyRelu':
			self.activ_func1 = activ_func(inplace=True)
		else:
			self.activ_func1 = activ_func()

		self.conv_2 = nn.Conv3d(num_channels, num_channels_out, (3, 3, 3), padding=padding, stride=stride)
		self.bn_2 = norm_layer2d(2, num_channels_out)
		if activ_func.__name__ == 'ReLU' or activ_func.__name__ == 'leakyRelu':
			self.activ_func2 = activ_func(inplace=True)
		else:
			self.activ_func2 = activ_func()

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
		:return: output tensor
		"""
		x = self.conv_1(input_tensor)
		x = self.bn_1(x)
		x = self.activ_func1(x)

		x = self.conv_2(x)
		x = self.bn_2(x)
		x = self.activ_func2(x)

		return x


# not working!
class CXRdecomp3(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		self.batch_size = batch_size
		self.enc_cna1 = DoubleConvBlock2d(input_views, 32, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna2 = DoubleConvBlock2d(32, 64, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna3 = DoubleConvBlock2d(64, 128, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna4 = DoubleConvBlock2d(128, 256, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna5 = DoubleConvBlock2d(256, 512, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna6 = DoubleConvBlock2d(512, 1024, use_norm=use_norm, activation_func=activation_func, stride=1)

		self.dec_cna1 = ConvNormActiv(in_channels=1024, out_channels=512, dim='2d', kernel_size=1, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)
		self.dec_cna2 = ConvNormActiv(in_channels=512, out_channels=256, dim='2d', kernel_size=1, use_norm='none', norm_layer=nn.Identity, activation_func='none')

	def forward(self, x_in):
		x = self.enc_cna1(x_in)
		x = self.enc_cna2(x)
		x = self.enc_cna3(x)
		x = self.enc_cna4(x)
		x = self.enc_cna5(x)
		x = self.enc_cna6(x)

		x = self.dec_cna1(x)
		out_d = self.dec_cna2(x)

		return torch.unsqueeze(out_d, 1)

# not working, too!
class CXRdecomp4(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		self.batch_size = batch_size
		self.cna1 = ConvNormActiv(in_channels=input_views, out_channels=32, dim='2d', kernel_size=1, use_norm=use_norm,
									  norm_layer=norm_layer2d, activation_func=activation_func)
		self.enc_cna1 = DoubleConvBlock2d(32, 64, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna2 = DoubleConvBlock2d(64, 128, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna3 = DoubleConvBlock2d(128, 256, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna4 = DoubleConvBlock2d(256, 512, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna5 = DoubleConvBlock2d(512, 1024, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna6 = DoubleConvBlock2d(1024, 2048, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.enc_cna7 = DoubleConvBlock2d(2048, 4096, use_norm=use_norm, activation_func=activation_func, stride=1)

		self.dec_cna1 = ConvNormActiv(in_channels=4096, out_channels=2048, dim='2d', kernel_size=1, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)
		self.dec_cna2 = ConvNormActiv(in_channels=2048, out_channels=1024, dim='2d', kernel_size=1, use_norm=use_norm, norm_layer=norm_layer2d, activation_func=activation_func)
		self.dec_cna3 = ConvNormActiv(in_channels=1024, out_channels=512, dim='2d', kernel_size=1, use_norm='none', norm_layer=nn.Identity, activation_func='none')
		self.dec_cna4 = ConvNormActiv(in_channels=512, out_channels=256, dim='2d', kernel_size=1, use_norm='none', norm_layer=nn.Identity, activation_func='none')

	def forward(self, x_in):
		x = self.cna1(x_in)
		x = self.enc_cna1(x)
		x = self.enc_cna2(x)
		x = self.enc_cna3(x)
		x = self.enc_cna4(x)
		x = self.enc_cna5(x)
		x = self.enc_cna6(x)
		x = self.enc_cna7(x)

		x = self.dec_cna1(x)
		x = self.dec_cna2(x)
		x = self.dec_cna3(x)
		out_d = self.dec_cna4(x)

		return torch.unsqueeze(out_d, 1)

# Works good
class CXRdecomp5(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		self.batch_size = batch_size
		self.input_views = input_views
		self.cna1 = ConvNormActiv(input_views, 128, activation_func=activation_func, norm_layer=norm_layer2d,
								  use_norm=use_norm, kernel_size=3, dim='2d')
		self.enc_cna1 = DoubleConvBlock2d(128, 128, use_norm=use_norm, activation_func=activation_func, stride=2)
		self.enc_cna2 = DoubleConvBlock2d(128, 256, use_norm=use_norm, activation_func=activation_func, stride=2)  # 128
		self.enc_cna3 = DoubleConvBlock2d(256, 512, use_norm=use_norm, activation_func=activation_func, stride=2)  # 64
		self.enc_cna4 = DoubleConvBlock2d(512, 1024, use_norm=use_norm, activation_func=activation_func, stride=2)  # 64
		self.enc_cna5 = DoubleConvBlock2d(1024, 2048, use_norm=use_norm, activation_func=activation_func, stride=2)  # 64

		self.trans_cna1 = ConvNormActiv(1024, 1024, activation_func=activation_func, norm_layer=norm_layer3d,
										use_norm=use_norm, kernel_size=1, dim='3d')
		self.trans_upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.trans_cna2 = ConvNormActiv(512, 512, activation_func=activation_func, norm_layer=norm_layer3d,
										use_norm=use_norm, kernel_size=1, dim='3d')
		self.trans_upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.trans_cna3 = ConvNormActiv(256, 256, activation_func=activation_func, norm_layer=norm_layer3d,
										use_norm=use_norm, kernel_size=1, dim='3d')
		self.trans_upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.dec_cna1 = DoubleConvBlock3d(256, 128, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.dec_cna2 = DoubleConvBlock3d(128, 64, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.dec_cna3 = DoubleConvBlock3d(64, 32, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_cna4 = ConvNormActiv(32, 1, activation_func='none', norm_layer=nn.Identity, use_norm=use_norm,
									  kernel_size=3, dim='3d')

	def forward(self, x_in):
		x = self.cna1(x_in)
		x = self.enc_cna1(x)
		x = self.enc_cna2(x)
		x = self.enc_cna3(x)
		x = self.enc_cna4(x)
		x = self.enc_cna5(x) # 2048, 8, 8

		x = x.reshape((x.shape[0], 1024, 2, 8, 8))
		x = self.trans_cna1(x)
		x = self.trans_upsample1(x)  # 1024, 4 ,16, 16

		x = x.reshape((x.shape[0], 512, 8, 16, 16))
		x = self.trans_cna2(x)
		x = self.trans_upsample2(x)  # 512, 16 ,32, 32

		x = x.reshape((x.shape[0], 256, 32, 32, 32))
		x = self.trans_cna3(x)
		x = self.trans_upsample3(x)  # 256, 64 ,64, 64

		x = self.dec_cna1(x)  # 128, 64 ,64, 64
		x = self.dec_upsample1(x)  # 128, 128 ,128, 128

		x = self.dec_cna2(x)  # 64, 128 ,128, 128
		x = self.dec_upsample2(x)  # 64, 256 ,256, 256

		x = self.dec_cna3(x)  # 32, 256 ,256, 256
		out_y = self.dec_cna4(x)  # 1, 256 ,256, 256

		return out_y

class CXRdecomp6(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		self.batch_size = batch_size
		self.input_views = input_views
		self.cna1 = ConvNormActiv(input_views, 128, activation_func=activation_func, norm_layer=norm_layer2d,
								  use_norm=use_norm, kernel_size=3, dim='2d')
		self.enc_cna1 = DoubleConvBlock2d(128, 128, use_norm=use_norm, activation_func=activation_func, stride=2) #->128
		self.enc_cna2 = DoubleConvBlock2d(128, 256, use_norm=use_norm, activation_func=activation_func, stride=2) #->64
		self.enc_cna3 = DoubleConvBlock2d(256, 512, use_norm=use_norm, activation_func=activation_func, stride=2) #->32
		self.enc_cna4 = DoubleConvBlock2d(512, 1024, use_norm=use_norm, activation_func=activation_func, stride=2) #->16
		#self.enc_cna5 = DoubleConvBlock2d(1024, 2048, use_norm=use_norm, activation_func=activation_func, stride=1) #->16

		self.trans_cna1 = ConvNormActiv(1024, 1024, activation_func=activation_func, norm_layer=norm_layer2d,
								  use_norm=use_norm, kernel_size=3, dim='2d')

		self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear') #-> 32
		self.dec_cna1 = DoubleConvBlock2d(1024 + 512, 1024, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear') #-> 64
		self.dec_cna2 = DoubleConvBlock2d(1024 + 256, 512, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample3 = nn.Upsample(scale_factor=2, mode='bilinear') #-> 128
		self.dec_cna3 = DoubleConvBlock2d(512 + 128, 512, use_norm=use_norm, activation_func=activation_func, stride=1)

		self.dec_upsample4 = nn.Upsample(scale_factor=2, mode='bilinear')  # -> 256
		self.dec_cna4 = DoubleConvBlock2d(512 , 512, use_norm=use_norm, activation_func=activation_func, stride=1)

		self.dec_cna5 = ConvNormActiv(512, 256, activation_func='none', norm_layer=nn.Identity, use_norm=use_norm,
									  kernel_size=1, dim='2d')


	def forward(self, x_in):
		x = self.cna1(x_in)
		down1 = self.enc_cna1(x)
		down2 = self.enc_cna2(down1)
		down3 = self.enc_cna3(down2)
		x = self.enc_cna4(down3)

		x = self.trans_cna1(x)

		x = self.dec_upsample1(x)
		x = torch.cat((x, down3), dim=1)
		x = self.dec_cna1(x)  # 128, 64 ,64, 64

		x = self.dec_upsample2(x)
		x = torch.cat((x, down2), dim=1)
		x = self.dec_cna2(x)  # 128, 64 ,64, 64

		x = self.dec_upsample3(x)
		x = torch.cat((x, down1), dim=1)
		x = self.dec_cna3(x)  # 128, 64 ,64, 64

		x = self.dec_upsample4(x)
		x = self.dec_cna4(x)  # 128, 64 ,64, 64

		out_y = torch.unsqueeze(self.dec_cna5(x), dim=1)  # 32, 256 ,256, 256

		return out_y

class CXRdecomp7(nn.Module):
	def __init__(self, batch_size, use_norm='group_norm', activation_func: str = 'relu', input_views=1):
		super().__init__()
		if use_norm == 'none':
			norm_layer2d = nn.Identity
			norm_layer3d = nn.Identity
		elif use_norm == 'group_norm':
			norm_layer2d = nn.GroupNorm
			norm_layer3d = nn.GroupNorm

		self.batch_size = batch_size
		self.input_views = input_views
		self.cna1 = ConvNormActiv(input_views, 128, activation_func=activation_func, norm_layer=norm_layer2d,
								  use_norm=use_norm, kernel_size=3, dim='2d')
		self.enc_cna1 = DoubleConvBlock2d(128, 128, use_norm=use_norm, activation_func=activation_func, stride=2) #->128
		self.enc_cna2 = DoubleConvBlock2d(128, 256, use_norm=use_norm, activation_func=activation_func, stride=2) #->64
		self.enc_cna3 = DoubleConvBlock2d(256, 512, use_norm=use_norm, activation_func=activation_func, stride=2) #->32
		self.enc_cna4 = DoubleConvBlock2d(512, 1024, use_norm=use_norm, activation_func=activation_func, stride=2) #->16
		self.enc_cna5 = DoubleConvBlock2d(1024, 2048, use_norm=use_norm, activation_func=activation_func, stride=2) #->8

		self.trans_cna1 = ConvNormActiv(2048, 2048, activation_func=activation_func, norm_layer=norm_layer2d,
								  use_norm=use_norm, kernel_size=3, dim='2d')

		self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear') #-> 16
		self.dec_cna1 = DoubleConvBlock2d(2048 + 1024, 2048, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear') #-> 32
		self.dec_cna2 = DoubleConvBlock2d(2048 + 512, 1024, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample3 = nn.Upsample(scale_factor=2, mode='bilinear') #-> 64
		self.dec_cna3 = DoubleConvBlock2d(1024 + 256, 512, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample4 = nn.Upsample(scale_factor=2, mode='bilinear')  # -> 128
		self.dec_cna4 = DoubleConvBlock2d(512 + 128, 512, use_norm=use_norm, activation_func=activation_func, stride=1)

		self.dec_upsample5 = nn.Upsample(scale_factor=2, mode='bilinear')  # -> 256
		self.dec_cna5 = DoubleConvBlock2d(512 , 512, use_norm=use_norm, activation_func=activation_func, stride=1)

		self.dec_cna6 = ConvNormActiv(512, 256, activation_func='none', norm_layer=norm_layer2d, use_norm=use_norm,
									  kernel_size=1, dim='2d')

	def forward(self, x_in):
		x = self.cna1(x_in)
		down1 = self.enc_cna1(x)
		down2 = self.enc_cna2(down1)
		down3 = self.enc_cna3(down2)
		down4 = self.enc_cna4(down3)
		x = self.enc_cna5(down4)

		x = self.trans_cna1(x)

		x = self.dec_upsample1(x)
		x = torch.cat((x, down4), dim=1)
		x = self.dec_cna1(x)  # 128, 64 ,64, 64

		x = self.dec_upsample2(x)
		x = torch.cat((x, down3), dim=1)
		x = self.dec_cna2(x)  # 128, 64 ,64, 64

		x = self.dec_upsample3(x)
		x = torch.cat((x, down2), dim=1)
		x = self.dec_cna3(x)  # 128, 64 ,64, 64

		x = self.dec_upsample4(x)
		x = torch.cat((x, down1), dim=1)
		x = self.dec_cna4(x)  # 128, 64 ,64, 64

		x = self.dec_upsample5(x)
		x = self.dec_cna5(x)  # 128, 64 ,64, 64

		out_y = torch.unsqueeze(self.dec_cna6(x), dim=1)  # 32, 256 ,256, 256

		return out_y