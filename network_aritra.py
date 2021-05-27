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


# Works good
class CXRdecomp(nn.Module):
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
		self.enc_cna1 = nn.MaxPool2d(2)
		#self.enc_cna1 = DoubleConvBlock2d(128, 128, use_norm=use_norm, activation_func=activation_func, stride=2)
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
		self.trans_cna4 = ConvNormActiv(128, 128, activation_func=activation_func, norm_layer=norm_layer3d,
										use_norm=use_norm, kernel_size=1, dim='3d')
		self.trans_upsample4 = nn.Upsample(scale_factor=2, mode='trilinear')

		self.dec_cna1 = DoubleConvBlock3d(128, 64, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.dec_cna2 = DoubleConvBlock3d(64, 32, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')
		self.dec_cna3 = DoubleConvBlock3d(32, 16, use_norm=use_norm, activation_func=activation_func, stride=1)
		self.dec_cna4 = ConvNormActiv(16, 1, activation_func='none', norm_layer=nn.Identity, use_norm=use_norm,
									  kernel_size=3, dim='3d')

		self.dec_cna5 = ConvNormActiv(512, 256, activation_func=activation_func, norm_layer=nn.Identity,
									  use_norm=use_norm,
									  kernel_size=3, dim='2d')
		self.dec_cna6 = ConvNormActiv(256, 3, activation_func='none', norm_layer=nn.Identity,
									  use_norm=use_norm,
									  kernel_size=3, dim='2d')

	def forward(self, x_in):
		x = self.cna1(x_in)
		x = self.enc_cna1(x)
		x = self.enc_cna2(x)
		x = self.enc_cna3(x)
		x = self.enc_cna4(x)
		x = self.enc_cna5(x) # 2048, 16, 16

		x = x.reshape((x.shape[0], 1024, 2, 16, 16))
		x = self.trans_cna1(x)
		x = self.trans_upsample1(x)  # 1024, 4 ,32, 32

		x = x.reshape((x.shape[0], 512, 8, 32, 32))
		x = self.trans_cna2(x)
		x = self.trans_upsample2(x)  # 512, 16 ,64, 64

		x = x.reshape((x.shape[0], 256, 32, 64, 64))
		x = self.trans_cna3(x)
		x = self.trans_upsample3(x)  # 256, 64 ,64, 64

		x = x.reshape((x.shape[0], 128, 128, 128, 128))
		x = self.trans_cna4(x)
		x = self.trans_upsample4(x)  # 128, 128 ,128, 128

		x = self.dec_cna1(x)  # 64, 128 ,128, 128
		x = self.dec_upsample1(x)  # 64, 256 ,256, 256

		x = self.dec_cna2(x)  # 32, 256 ,256, 256
		#x = self.dec_upsample2(x)  # 32, 512 ,512, 512

		x = self.dec_cna3(x)  # 16, 512 ,512, 512
		out_y = self.dec_cna4(x)  # 1, 512 ,512, 512

		x = x.reshape((out_y.shape[0], 512, 512, 512))
		x = self.dec_cna5(x)  #  256 ,512, 512
		out_yy = self.dec_cna6(x)  #  3 ,512, 512

		return out_y, out_yy


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

		self.dec_cna5 = ConvNormActiv(256, 128, activation_func=activation_func, norm_layer=nn.Identity,
									  use_norm=use_norm,
									  kernel_size=3, dim='2d')
		self.dec_cna6 = ConvNormActiv(128, input_views, activation_func='none', norm_layer=nn.Identity,
									  use_norm=use_norm,
									  kernel_size=3, dim='2d')

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

		x = out_y.reshape((out_y.shape[0], 256, 256, 256))
		x = self.dec_cna5(x)
		out_yy = self.dec_cna6(x)

		return out_y, out_yy
