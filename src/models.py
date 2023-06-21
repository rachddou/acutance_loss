"""
Definition of the FFDNet model and its custom layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch.nn as nn
from torch.autograd import Variable
from  src.functions import concatenate_input_noise_map,upsamplefeatures

class UpSampleFeatures(nn.Module):
	r"""Implements the last layer of FFDNet
	"""
	def __init__(self):
		super(UpSampleFeatures, self).__init__()
	def forward(self, x):
		return upsamplefeatures(x)

class IntermediateDnCNN(nn.Module):
	r"""Implements the middle part of the FFDNet architecture, which
	is basically a DnCNN net
	"""
	def __init__(self, input_features, middle_features, num_conv_layers):
		super(IntermediateDnCNN, self).__init__()
		self.kernel_size = 3
		self.padding = 1
		self.input_features = input_features
		self.num_conv_layers = num_conv_layers
		self.middle_features = middle_features
		if self.input_features == 5:
			self.output_features = 4 #Grayscale image
		elif self.input_features == 15:
			self.output_features = 12 #RGB image
		else:
			raise Exception('Invalid number of input features')

		layers = []
		layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(self.num_conv_layers-2):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		self.itermediate_dncnn = nn.Sequential(*layers)
	def forward(self, x):
		out = self.itermediate_dncnn(x)
		return out

class FFDNet(nn.Module):
	r"""Implements the FFDNet architecture
	"""
	def __init__(self, num_input_channels):
		super(FFDNet, self).__init__()
		self.num_input_channels = num_input_channels
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = IntermediateDnCNN(\
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()

	def forward(self, x, noise_map):
		concat_noise_x = concatenate_input_noise_map(\
				x.data, noise_map.data)
		concat_noise_x = Variable(concat_noise_x)
		h_dncnn = self.intermediate_dncnn(concat_noise_x)
		pred_img = self.upsamplefeatures(h_dncnn)
		return pred_img

class FFDNet_fine_tuning(nn.Module):
	r"""Implements the FFDNet architecture
	"""
	def __init__(self, num_input_channels):
		super(FFDNet_fine_tuning, self).__init__()
		self.num_input_channels = num_input_channels
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = IntermediateDnCNN(\
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()
		for params in list(self.intermediate_dncnn.parameters())[0:50]:
			params.requires_grad = False   
	def forward(self, x, noise_sigma):
		concat_noise_x = concatenate_input_noise_map(\
				x.data, noise_sigma.data)
		concat_noise_x = Variable(concat_noise_x)
		h_dncnn = self.intermediate_dncnn(concat_noise_x)
		pred_img = self.upsamplefeatures(h_dncnn)
		return pred_img

class Discriminator(nn.Module):
	r"""Implements the FFDNet Discriminator
	"""
	def __init__(self,num_input_channels):
		super(Discriminator, self,).__init__()
		self.kernel_size = 3
		self.padding = 1
		self.input_features = num_input_channels
		# self.num_input_channels = num_input_channels
		conv_layers = []
		linear_layers = []
		#requires size 64 or 50 as input
		conv_layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=32,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		conv_layers.append(nn.Conv2d(in_channels=32,\
								out_channels=64,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								stride = (2,2),\
								bias=False))
		conv_layers.append(nn.Conv2d(in_channels=64,\
								out_channels=64,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		conv_layers.append(nn.Conv2d(in_channels=64,\
								out_channels=128,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								stride = (2,2),\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		conv_layers.append(nn.Conv2d(in_channels=128,\
								out_channels=128,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		conv_layers.append(nn.Conv2d(in_channels=128,\
								out_channels=256,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								stride = (2,2),\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		conv_layers.append(nn.Conv2d(in_channels=256,\
								out_channels=256,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		conv_layers.append(nn.Conv2d(in_channels=256,\
								out_channels=512,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								stride = (2,2),\
								bias=False))
		conv_layers.append(nn.ReLU(inplace=True))
		linear_layers.append(nn.Linear(in_features = 8192, out_features = 1024 ))
		linear_layers.append(nn.ReLU(inplace=True))
		linear_layers.append(nn.Linear(in_features =1024, out_features = 1 ))
		linear_layers.append(nn.Sigmoid())
		self.conv_model = nn.Sequential(*conv_layers)		
		self.linear_model = nn.Sequential(*linear_layers)


	def forward(self, x):
		feature = self.conv_model(x)
		feature = feature.view(feature.size(0), -1)
		validity = self.linear_model(feature)
		return validity